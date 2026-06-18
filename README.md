# ThinkTwice · 推理时不确定性检测

当前为预览版本，非正式版，完整报告见https://gmunitx.com/index.php/thinktwice

**让语言模型在不确定时诚实地表达边界，而非强行编造。**

ThinkTwice 是一个轻量级的推理增强框架，无需重新训练或微调，即可显著降低大模型的幻觉率。它通过监控注意力模式变化与多路径分歧检测，使模型在不确定时主动拒答或表达不确定性，从而提升 AI 系统的诚实性与可信度。

---

## 🧠 核心思路

- **注意力相似度 → 步骤边界检测**  
  在自回归生成过程中，ThinkTwice 实时监测注意力向量的演变。当模型完成一个推理步骤并转向下一阶段时，注意力分布会发生可量化的“断裂”，此信号用于切分步骤。

- **分歧引导的自省机制**  
  维护多条并行推理路径，交叉验证输出一致性。若不同路径在步骤尾部出现显著分歧，框架自动触发自检提示，引导模型重新评估或直接拒答，若头部产生分歧，视为创造性分歧，保护模型创造力。

- **无需重训练，即插即用**  
  不修改模型权重，不依赖外部知识库。仅需在推理阶段增加轻量化监控层，即可为LLM增强诚实性。

---

## 📊 评测结果

### AA‑Omniscience（600 题，6 大领域）

| 指标 | 数值 |
|------|------|
| ✅ 正确 | 19 |
| 🔸 部分正确 | 2 |
| ❌ 错误 | 134 |
| ⚪ 拒答 | 445 |
| **准确率** | 3.17% |
| **幻觉率** | 23.14% （错误 / (错误+拒答)） |
| **全知指数** | -19.2 |

### TruthfulQA（30 题，对抗性误导）

| 指标 | ThinkTwice |
|------|------------|
| 总分 | 19.90 / 30 |
| 严重幻觉数 | 7（较基线 ↓36%） |
| 平均分 | 0.663 |

> 模型在易产生“死亡幻觉”的年龄问题、历史时序等陷阱上，主动中断推理或标注不确定性。

详细评测日志和评分标准见https://gmunitx.com/index.php/thinktwice

---

注：目前处于实验阶段，代码稳健性不足，可能不兼容许多语言模型，目前已知的兼容性问题有：1.预设对话模板提取方式导致使用非ChatML模板的模型不直接兼容，使用前须修改代码；2.不兼容深度思考模型，但可以用非深度思考模式使用（可开关深度思考的模型）；3.未测试过多模态模型。建议使用Qwen2.5系列语言模型快速开始测试。

# ThinkTwice · Uncertainty Detection During Inference

> *Preview version, not final release. Full report: https://gmunitx.com/index.php/thinktwice*

**Let language models honestly express their boundaries when uncertain, rather than fabricating answers.**

ThinkTwice is a lightweight inference‑enhancement framework that significantly reduces hallucination rates in large language models **without retraining or fine‑tuning**. By monitoring changes in attention patterns and multi‑path divergence, it enables models to actively refuse answers or express uncertainty when they are unsure, thereby improving honesty and trustworthiness in AI systems.

---

## 🧠 Core Ideas

- **Attention Similarity → Step‑Boundary Detection**  
  During autoregressive generation, ThinkTwice monitors the evolution of attention vectors in real time. When the model completes one reasoning step and transitions to the next, the attention distribution exhibits a quantifiable "break"—this signal is used to segment steps.

- **Divergence‑Driven Self‑Reflection**  
  Multiple parallel reasoning paths are maintained to cross‑validate output consistency. If significant divergence appears at the *tail* of a step, the framework automatically triggers a self‑check prompt, guiding the model to re‑evaluate or directly refuse to answer. Divergence at the *head* is treated as creative divergence, preserving the model's creativity.

- **No Retraining, Plug‑and‑Play**  
  No model weights are modified, and no external knowledge bases are required. Only a lightweight monitoring layer is added during inference, making LLMs more honest out of the box.

---

## 📊 Evaluation Results

### AA‑Omniscience (600 questions, 6 domains)

| Metric | Value |
|--------|-------|
| ✅ Correct | 19 |
| 🔸 Partially Correct | 2 |
| ❌ Incorrect | 134 |
| ⚪ Refused | 445 |
| **Accuracy** | 3.17% |
| **Hallucination Rate** | 23.14% （Incorrect / (Incorrect + Refused)） |
| **Omniscience Index** | -19.2 |

### TruthfulQA (30 questions, adversarial misleading)

| Metric | ThinkTwice |
|--------|------------|
| Total Score | 19.90 / 30 |
| Severe Hallucinations | 7 （↓36% vs baseline） |
| Average Score | 0.663 |

> On trap questions that tend to cause "mortality hallucinations"—such as age‑related questions and historical chronology—the model proactively interrupts reasoning or annotates uncertainty.

Detailed evaluation logs and scoring criteria are available at:  
https://gmunitx.com/index.php/thinktwice

---

**Note:** This is currently experimental; code robustness is limited and may not be compatible with many language models. Known compatibility issues include:
1. The preset chat‑template extraction method may not work with models that do not use the ChatML template—code modification is required before use.
2. Does not support deep‑thinking models, but can be used in non‑deep‑thinking mode (for models with switchable deep‑thinking).
3. Not yet tested with multimodal models.

We recommend starting with the **Qwen2.5** series of language models for quick testing.
