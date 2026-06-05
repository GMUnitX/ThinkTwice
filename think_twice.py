# think_twice.py
# By GMUnitX.
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastdtw import fastdtw
from typing import List, Dict, Tuple, Optional, Generator
import copy
import config
from concurrent.futures import ThreadPoolExecutor, as_completed

class ThinkTwiceFramework:
    def __init__(self):
        self.device = None
        self.model = None
        self.tokenizer = None
        self.stress = config.STRESS_INITIAL
        self.global_step = 0
        self._load_model()

    def _load_model(self):
        """加载模型和分词器，自动识别设备"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME_OR_PATH,
            trust_remote_code=config.TRUST_REMOTE_CODE
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = config.TORCH_DTYPE
        if torch_dtype == "auto":
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            torch_dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

        model_kwargs = {
            "device_map": config.DEVICE_MAP,
            "dtype": torch_dtype,
            "trust_remote_code": config.TRUST_REMOTE_CODE,
            "attn_implementation": "eager",  # 必须使用 eager 以输出注意力权重
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME_OR_PATH,
            **model_kwargs
        )
        self.model.eval()
        if hasattr(self.model, 'device'):
            self.device = self.model.device
        else:
            self.device = next(self.model.parameters()).device

    def _format_conversation(self, conversation: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """使用模型的对话模板格式化对话历史"""
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
        else:
            # 回退到简单拼接
            formatted = []
            for msg in conversation:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    formatted.append(f"system: {content}")
                elif role == "user":
                    formatted.append(f"user: {content}")
                elif role == "assistant":
                    formatted.append(f"assistant: {content}")
                else:
                    formatted.append(content)
            if add_generation_prompt:
                formatted.append("assistant:")
            return "\n".join(formatted)

    def generate(self, conversation: List[Dict[str, str]]) -> Generator[str, None, None]:
        """主生成入口，流式输出生成的文本 token"""
        current_conversation = copy.deepcopy(conversation)
        if not current_conversation or current_conversation[-1].get("role") != "assistant":
            current_conversation.append({"role": "assistant", "content": ""})

        formatted_prompt = self._format_conversation(current_conversation, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_prompt, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        past_key_values = None
        total_generated = 0
        max_total_tokens = config.MAX_NEW_TOKENS
        finished = False

        # 预填充：获取初始 KV 缓存
        if past_key_values is None:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False
                )
                past_key_values = outputs.past_key_values

        self.generation_log = []

        while total_generated < max_total_tokens and not finished:
            remaining = max_total_tokens - total_generated
            step_result = self._run_step(
                past_key_values=past_key_values,
                current_token=input_ids[:, -1:],
                current_conversation=current_conversation,
                remaining_tokens=remaining
            )

            if step_result is None:
                break

            new_tokens_ids = step_result['new_tokens']
            if len(new_tokens_ids) == 0:
                break

            for token_id in new_tokens_ids:
                text = self.tokenizer.decode(token_id, skip_special_tokens=True)
                yield text

            new_text = self.tokenizer.decode(new_tokens_ids, skip_special_tokens=True)
            current_conversation[-1]["content"] += new_text

            past_key_values = step_result['kv_cache']
            total_generated += len(new_tokens_ids)
            input_ids = torch.cat([input_ids, new_tokens_ids.unsqueeze(0)], dim=1)

            if step_result.get('terminate', False):
                finished = True

    def _run_step(self, past_key_values, current_token: torch.Tensor,
                  current_conversation: List[Dict[str, str]], remaining_tokens: int) -> Optional[Dict]:
        """执行一轮 Think Twice 步骤（并行推理）"""
        n_paths = config.NUM_PATHS
        path_results = [None] * n_paths  # 预分配，保持顺序
        step_log = {'paths': [], 'stress_before': self.stress, 'decision': None}

        # 使用线程池并行执行多条路径
        with ThreadPoolExecutor(max_workers=n_paths) as executor:
            future_to_idx = {}
            for path_idx in range(n_paths):
                future = executor.submit(
                    self._run_single_path,
                    past_key_values=past_key_values,
                    start_token=current_token,
                    path_idx=path_idx,
                    remaining_tokens=remaining_tokens
                )
                future_to_idx[future] = path_idx

            # 收集结果，按完成顺序处理（但最终按索引排序）
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    path_info = future.result()
                    path_results[idx] = path_info
                except Exception as e:
                    print(f"路径 {idx} 生成异常: {e}")
                    # 构造一个表示提前终止的占位结果
                    path_results[idx] = {
                        'path_idx': idx,
                        'tokens': torch.tensor([], device=self.device),
                        'confidences': [],
                        'attn_similarities': [],
                        'hidden_states': [],
                        'path_confidence': 0.0,
                        'terminated_early': True,
                        'reached_step_boundary': False,
                        'kv_cache': None
                    }

        # 整理日志信息
        for path_info in path_results:
            step_log['paths'].append({
                'path_idx': path_info['path_idx'],
                'tokens': path_info['tokens'].tolist() if path_info['tokens'] is not None else [],
                'confidences': path_info['confidences'],
                'attn_similarities': path_info['attn_similarities'],
                'path_confidence': path_info['path_confidence'],
                'terminated_early': path_info['terminated_early'],
                'reached_step_boundary': path_info['reached_step_boundary'],
            })

        # 过滤有效路径（未被提前终止的）
        valid_paths = [p for p in path_results if not p['terminated_early']]

        # 所有路径均失败：生成致歉并终止
        if len(valid_paths) == 0:
            apology_ids, _ = self._generate_apology(current_conversation, remaining_tokens)
            if apology_ids is not None and apology_ids.shape[1] > 0:
                step_log['decision'] = 'all_paths_failed_apology'
                if config.VERBOSE:
                    self._print_step_log(step_log)
                return {
                    'new_tokens': apology_ids[0],
                    'kv_cache': past_key_values,
                    'terminate': True
                }
            else:
                return None

        # 按路径置信度排序，取前两条
        valid_paths.sort(key=lambda x: x['path_confidence'], reverse=True)
        top_two = valid_paths[:2]

        # 只有一条有效路径：直接采纳，无分歧
        if len(top_two) == 1:
            chosen_path = top_two[0]
            step_log['decision'] = 'single_path'
            if config.VERBOSE:
                self._print_step_log(step_log)
            terminate = (len(chosen_path['tokens']) > 0 and 
                         chosen_path['tokens'][-1].item() == self.tokenizer.eos_token_id)
            return {
                'new_tokens': chosen_path['tokens'],
                'kv_cache': chosen_path['kv_cache'],
                'terminate': terminate
            }

        # 分歧检测
        path_a, path_b = top_two[0], top_two[1]
        hidden_a = path_a['hidden_states']
        hidden_b = path_b['hidden_states']

        if len(hidden_a) == 0 or len(hidden_b) == 0:
            chosen_path = path_a
            step_log['decision'] = 'empty_hidden_fallback'
            if config.VERBOSE:
                self._print_step_log(step_log)
            terminate = (len(chosen_path['tokens']) > 0 and 
                         chosen_path['tokens'][-1].item() == self.tokenizer.eos_token_id)
            return {
                'new_tokens': chosen_path['tokens'],
                'kv_cache': chosen_path['kv_cache'],
                'terminate': terminate
            }

        seq_a = torch.stack(hidden_a).cpu().numpy()
        seq_b = torch.stack(hidden_b).cpu().numpy()

        def cosine_distance(x, y):
            sim = F.cosine_similarity(torch.tensor(x).unsqueeze(0), torch.tensor(y).unsqueeze(0), dim=1).item()
            return 1.0 - sim

        distance, path = fastdtw(seq_a, seq_b, radius=config.FASTDTW_RADIUS, dist=cosine_distance)

        aligned_a = np.array([seq_a[i] for i, _ in path])
        aligned_b = np.array([seq_b[j] for _, j in path])

        total_len = len(aligned_a)
        head_len = int(total_len * config.HEAD_RATIO)
        head_a = aligned_a[:head_len]
        head_b = aligned_b[:head_len]
        tail_a = aligned_a[head_len:]
        tail_b = aligned_b[head_len:]

        def avg_cosine_sim(a, b):
            if len(a) == 0:
                return 1.0
            sims = [F.cosine_similarity(torch.tensor(x).unsqueeze(0), torch.tensor(y).unsqueeze(0), dim=1).item()
                    for x, y in zip(a, b)]
            return np.mean(sims)

        head_sim = avg_cosine_sim(head_a, head_b)
        tail_sim = avg_cosine_sim(tail_a, tail_b) if len(tail_a) > 0 else 1.0

        chosen_path = None
        divergence_type = None
        if head_sim < config.HEAD_SIMILARITY_THRESHOLD:
            divergence_type = 'creative'
            chosen_path = path_a
        elif tail_sim < config.TAIL_SIMILARITY_THRESHOLD:
            divergence_type = 'error'
            chosen_path = path_a
        else:
            divergence_type = 'none'
            chosen_path = path_a

        terminate = False

        # ==================== 修改点：错误分歧处理 ====================
        if divergence_type == 'error':
            self.stress += (1.0 - tail_sim) * config.STRESS_INCREASE_FACTOR

            # 丢弃当前路径生成的所有 token，基于步骤开始时的状态生成自检/道歉
            # 使用原始的 past_key_values（步骤开始时的 KV 缓存）
            original_past_key_values = past_key_values
            remaining_for_check = remaining_tokens

            # 生成自检信息
            self_check_ids, self_check_text = self._generate_self_check(
                current_conversation, remaining_for_check
            )
            new_tokens_list = []
            current_kv = original_past_key_values

            if self_check_ids is not None and self_check_ids.shape[1] > 0:
                # 将自检 token 通过模型前向，获得新的 KV Cache
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=self_check_ids,
                        past_key_values=current_kv,
                        use_cache=True
                    )
                current_kv = outputs.past_key_values
                new_tokens_list.append(self_check_ids[0])  # shape (num_tokens,)
                remaining_for_check -= self_check_ids.shape[1]

            # 若 Stress 超过阈值，生成道歉信息（追加在自检之后）
            if self.stress >= config.STRESS_THRESHOLD:
                # 临时构造包含自检文本的对话（用于生成道歉）
                temp_conv_for_apology = copy.deepcopy(current_conversation)
                if self_check_text:
                    temp_conv_for_apology[-1]["content"] += self_check_text
                apology_ids, apology_text = self._generate_apology(
                    temp_conv_for_apology, remaining_for_check
                )
                if apology_ids is not None and apology_ids.shape[1] > 0:
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=apology_ids,
                            past_key_values=current_kv,
                            use_cache=True
                        )
                    current_kv = outputs.past_key_values
                    new_tokens_list.append(apology_ids[0])
                terminate = True  # 道歉后结束当前生成轮次

            # 将多个 token 段拼接成一个张量
            if new_tokens_list:
                final_tokens = torch.cat(new_tokens_list, dim=0)
            else:
                # 如果自检和道歉都失败，返回空（上层会处理）
                final_tokens = torch.tensor([], device=self.device, dtype=torch.long)

            step_log['decision'] = 'error_apology' if terminate else 'error_self_check'
            if config.VERBOSE:
                self._print_step_log(step_log)

            return {
                'new_tokens': final_tokens,
                'kv_cache': current_kv,
                'terminate': terminate
            }
        # ===========================================================

        # 非错误分歧（creative 或 none）的处理
        if divergence_type == 'creative':
            self.stress = max(0.0, self.stress - config.STRESS_DECREASE_STEP)
        else:
            self.stress = max(0.0, self.stress - config.STRESS_DECREASE_STEP)

        if len(chosen_path['tokens']) > 0 and chosen_path['tokens'][-1].item() == self.tokenizer.eos_token_id:
            terminate = True

        step_log['decision'] = f'chose_path_{chosen_path["path_idx"]}_divergence_{divergence_type}'
        if config.VERBOSE:
            self._print_step_log(step_log)

        return {
            'new_tokens': chosen_path['tokens'],
            'kv_cache': chosen_path['kv_cache'],
            'terminate': terminate
        }

    def _apply_repetition_penalty(self, logits: torch.Tensor, generated_ids: List[int]) -> torch.Tensor:
        """对已生成的 token 应用重复惩罚"""
        if config.REPETITION_PENALTY == 1.0 or not generated_ids:
            return logits

        penalty = config.REPETITION_PENALTY
        score = torch.gather(logits, 1, torch.tensor(generated_ids, device=self.device).unsqueeze(0))
        score = torch.where(score > 0, score / penalty, score * penalty)
        logits.scatter_(1, torch.tensor(generated_ids, device=self.device).unsqueeze(0), score)
        return logits

    def _run_single_path(self, past_key_values, start_token: torch.Tensor,
                         path_idx: int, remaining_tokens: int) -> Dict:
        """运行单条生成路径，包含重复惩罚和KV缓存处理（线程安全）"""
        # --- 健壮的 KV 缓存深拷贝，确保线程间隔离 ---
        if past_key_values is not None:
            try:
                past_key_values = copy.deepcopy(past_key_values)
            except Exception:
                if isinstance(past_key_values, tuple):
                    pkv = []
                    for layer in past_key_values:
                        if layer is None:
                            pkv.append(None)
                        else:
                            pkv.append(tuple(t.clone() for t in layer))
                    past_key_values = tuple(pkv)
                else:
                    try:
                        pkv = []
                        for layer in past_key_values:
                            if hasattr(layer, 'key') and hasattr(layer, 'value'):
                                pkv.append((layer.key.clone(), layer.value.clone()))
                            elif isinstance(layer, (tuple, list)) and len(layer) == 2:
                                pkv.append((layer[0].clone(), layer[1].clone()))
                            else:
                                raise AttributeError
                        past_key_values = tuple(pkv)
                    except (TypeError, AttributeError):
                        if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
                            pkv = []
                            for layer_idx in range(len(past_key_values.key_cache)):
                                key = past_key_values.key_cache[layer_idx].clone()
                                value = past_key_values.value_cache[layer_idx].clone()
                                pkv.append((key, value))
                            past_key_values = tuple(pkv)
                        else:
                            print(f"警告：无法深拷贝 KV 缓存（类型 {type(past_key_values)}），路径间可能相互影响。")

        generated_ids = []
        confidences = []
        attn_similarities = []
        hidden_states = []
        terminated_early = False
        reached_step_boundary = False

        current_token = start_token
        prev_attn_vector = None

        for step in range(remaining_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True,
                    output_hidden_states=True
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            logits = self._apply_repetition_penalty(logits, generated_ids)

            if config.TEMPERATURE != 1.0:
                logits = logits / config.TEMPERATURE

            if config.TOP_K > 0:
                indices_to_remove = logits < torch.topk(logits, config.TOP_K)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            if config.TOP_P < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > config.TOP_P
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            max_prob = probs.max().item()
            confidences.append(max_prob)

            if max_prob < config.CONFIDENCE_THRESHOLD and len(generated_ids) >= config.MIN_PATH_LENGTH:
                terminated_early = True
                break

            generated_ids.append(next_token_id.item())

            last_hidden = outputs.hidden_states[-1][0, -1, :]
            hidden_states.append(last_hidden)

            # 注意力相似度计算
            attn_weights = outputs.attentions[config.ATTENTION_LAYER_INDEX]
            avg_attn = attn_weights.mean(dim=1)  # (1, query_len, seq_len)
            curr_attn_vector = avg_attn[0, 0, :].clone()  # 形状 (seq_len,)
            current_seq_len = curr_attn_vector.shape[0]

            if prev_attn_vector is not None:
                common_len = current_seq_len - 1
                if common_len > 0:
                    v_curr = curr_attn_vector[:common_len]
                    v_prev = prev_attn_vector[:common_len]
                    if v_curr.numel() > 0 and v_prev.numel() > 0:
                        sim = F.cosine_similarity(v_curr.unsqueeze(0), v_prev.unsqueeze(0)).item()
                    else:
                        sim = 1.0
                else:
                    sim = 1.0
                attn_similarities.append(sim)

                if config.VERBOSE:
                    print(f"  [Path {path_idx}] Token {len(generated_ids)}: common_len={common_len}, sim={sim:.4f}")

                if sim < config.ATTENTION_SIMILARITY_THRESHOLD:
                    reached_step_boundary = True
                    break
            else:
                attn_similarities.append(1.0)

            prev_attn_vector = curr_attn_vector
            current_token = next_token_id

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        path_confidence = np.mean(confidences) if confidences else 0.0

        return {
            'path_idx': path_idx,
            'tokens': torch.tensor(generated_ids, device=self.device),
            'confidences': confidences,
            'attn_similarities': attn_similarities,
            'hidden_states': hidden_states,
            'path_confidence': path_confidence,
            'terminated_early': terminated_early,
            'reached_step_boundary': reached_step_boundary,
            'kv_cache': past_key_values if not terminated_early else None
        }

    def _build_check_message_conversation(self, base_conversation: List[Dict[str, str]], instruction: str) -> List[Dict[str, str]]:
        """构造用于生成自检/致歉信息的新对话（将历史嵌入用户消息）"""
        system_msg = None
        for msg in base_conversation:
            if msg.get("role") == "system":
                system_msg = msg
                break

        conv_text = self._format_conversation(base_conversation, add_generation_prompt=False)

        new_conv = []
        if system_msg:
            new_conv.append(system_msg.copy())
        else:
            new_conv.append({"role": "system", "content": "你是一个帮助生成对话衔接内容的助手。"})
        
        user_content = f"{instruction}\n\n对话历史：\n{conv_text}"
        new_conv.append({"role": "user", "content": user_content})
        new_conv.append({"role": "assistant", "content": ""})
        return new_conv

    def _generate_self_check(self, conversation: List[Dict[str, str]], remaining_tokens: int) -> Tuple[Optional[torch.Tensor], str]:
        """生成自检信息（表示困惑的自然衔接）"""
        if remaining_tokens <= 0:
            return None, ""
        instruction = config.SELF_CHECK_PROMPT_TEMPLATE
        new_conv = self._build_check_message_conversation(conversation, instruction)
        
        prompt = self._format_conversation(new_conv, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        max_new = min(config.SELF_CHECK_MAX_TOKENS, remaining_tokens)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            top_k=config.TOP_K,
            repetition_penalty=config.REPETITION_PENALTY,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        if generated_ids.shape[1] > 0 and generated_ids[0, -1].item() == self.tokenizer.eos_token_id:
            generated_ids = generated_ids[:, :-1]
        if generated_ids.shape[1] == 0:
            return None, ""
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_ids, text

    def _generate_apology(self, conversation: List[Dict[str, str]], remaining_tokens: int) -> Tuple[Optional[torch.Tensor], str]:
        """生成致歉信息（表示能力不足并结束对话）"""
        if remaining_tokens <= 0:
            return None, ""
        instruction = config.APOLOGY_PROMPT_TEMPLATE
        new_conv = self._build_check_message_conversation(conversation, instruction)
        
        prompt = self._format_conversation(new_conv, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        max_new = min(config.APOLOGY_MAX_TOKENS, remaining_tokens)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            top_k=config.TOP_K,
            repetition_penalty=config.REPETITION_PENALTY,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        if generated_ids.shape[1] > 0 and generated_ids[0, -1].item() == self.tokenizer.eos_token_id:
            generated_ids = generated_ids[:, :-1]
        if generated_ids.shape[1] == 0:
            return None, ""
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_ids, text

    def _print_step_log(self, step_log: Dict):
        """详细模式打印步骤日志"""
        print(f"\n==== Step {self.global_step} ====")
        print(f"Stress before: {step_log['stress_before']:.3f}")
        for path in step_log['paths']:
            tokens_str = self.tokenizer.decode(path['tokens']) if path['tokens'] else ""
            print(f"Path {path['path_idx']}: tokens={tokens_str}, confidences={path['confidences']}, "
                  f"attn_sims={path['attn_similarities']}, path_conf={path['path_confidence']:.3f}, "
                  f"terminated={path['terminated_early']}, boundary={path['reached_step_boundary']}")
        print(f"Decision: {step_log['decision']}")
        print(f"Stress after: {self.stress:.3f}")
        self.global_step += 1