# think_twice/config.py
"""
Think Twice 框架配置文件
所有可调参数均在此处集中管理，包含详细注释说明
"""

# ==================== 模型加载配置 ====================
MODEL_NAME_OR_PATH = "./Qwen2.5-3B-Instruct"  # 模型名称或本地路径
DEVICE_MAP = "auto"                                   # 设备映射，"auto" 自动识别 CPU/GPU
TORCH_DTYPE = "auto"                                  # 模型数据类型，"auto" 或 torch.float16 等
TRUST_REMOTE_CODE = False                             # 是否信任远程代码（某些自定义模型需要）
USE_FLASH_ATTENTION = False                           # 是否使用 Flash Attention（需要安装）

# ==================== 推理参数配置 ====================
NUM_PATHS = 4                                         # 并行路径数量 n
TEMPERATURE = 0.7                                     # 温度参数，控制随机性
TOP_P = 0.9                                           # nucleus 采样阈值
TOP_K = 0                                            # top-k 采样，0 表示不限制
MAX_NEW_TOKENS = 40960                                  # 整个请求的最大生成 token 数
REPETITION_PENALTY = 1.05                              # 重复惩罚系数

# ==================== 注意力监测与步骤边界 ====================
ATTENTION_SIMILARITY_THRESHOLD = 0.8                  # 注意力相似度低于该值时视为步骤边界
ATTENTION_LAYER_INDEX = -1                            # 使用哪一层的注意力，-1 表示最后一层

# ==================== 置信度与路径早停 ====================
CONFIDENCE_THRESHOLD = 0.2                            # token 置信度低于该值时提前终止路径
MIN_PATH_LENGTH = 3                                   # 路径至少生成多少个 token 后才允许置信度过低终止

# ==================== 分歧检测配置 ====================
HEAD_RATIO = 0.35                                      # 头部占比（用于计算头部相似度）
HEAD_SIMILARITY_THRESHOLD = 0.8                       # 头部相似度低于该值时视为创造性分歧
TAIL_SIMILARITY_THRESHOLD = 0.7                       # 尾部相似度低于该值时（且头部不低）视为错误性分歧
FASTDTW_RADIUS = 1                                    # FastDTW 的 radius 参数，控制搜索窗口大小

# ==================== 压力值机制配置 ====================
STRESS_INITIAL = 0.0                                  # 初始压力值
STRESS_THRESHOLD = 64.0                                # 压力值超过该阈值时致歉终止
STRESS_DECREASE_STEP = 1.0                            # 无分歧或创造性分歧时压力降低量
STRESS_INCREASE_FACTOR = 1.2                          # 错误性分歧时压力增加因子（乘上 1-尾部相似度）

# ==================== 自检与致歉信息生成 ====================
# 注意：以下模板将在代码中动态填充对话历史，格式为 "system:\n...\nuser:\n...\nassistant:..."
SELF_CHECK_PROMPT_TEMPLATE = (
    "{conversation}\n"
    "请你根据以上内容，帮助助手生成一个可以衔接在助手输出的话的尾部的明确提醒用户并拒绝继续回答的信息，并确保不包含多余的信息。"
    "例如：“……不太对……我觉得我前面的回答是错误的，我可能不太记得相关信息了，请您注意核实。如果有什么其他需要帮助的，请随时告诉我。若您希望我重新回想，也可以随时告知。”"
    "根据实际情况生成，例如知识性问题就表明不知道，计算性问题就怀疑算错了，不要给出你认为的信息（即不要替助手回答相关信息），仅按照大类别给出语句，不要具体分析是怎么错的。"
)

APOLOGY_PROMPT_TEMPLATE = (
    "{conversation}\n"
    "请你根据以上内容，帮助助手生成一个可以衔接在助手输出的话的尾部的明确提醒用户并拒绝继续回答，并确保不包含多余的信息。"
    "例如：“……好像不太对劲……抱歉，我的能力不足以解决这个问题，我前面的回答是错误的，我可能不太记得详细信息了。如果有什么其他需要帮助的，请随时告诉我。若您希望我重新回想，也可以随时告知。”"
    "根据实际情况生成，例如知识性问题就表明不知道，计算性问题就怀疑算错了，不要给出你认为的信息（即不要替助手回答相关信息），仅按照大类别给出语句，不要具体分析是怎么错的。"
)

SELF_CHECK_MAX_TOKENS = 4096                            # 自检信息最大生成 token 数（实际取 min(此值, 剩余token)）
APOLOGY_MAX_TOKENS = 4096                               # 致歉信息最大生成 token 数（实际取 min(此值, 剩余token)）

# ==================== 输出与调试配置 ====================
VERBOSE = False                                       # 详细模式，输出每个路径的 token、置信度、相似度等
STREAMING = True                                      # 是否流式输出（本框架实现为逐步 yield token）