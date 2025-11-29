import torch
import sys 

# --- 核心配置 ---
# 数据文件路径
DATA_PATH = "emotion_data_manual.csv"
# 训练结果和检查点保存的基础目录
CACHE_PATH = "results_18emo"
# 最终模型的保存目录
FINAL_MODEL_DIR = "./emotion_model_18emo"
# 随机种子
SEED = 42

# 明确定义18类情绪及其顺序 (关键！)
TARGET_EMOTIONS = [
    "高兴", "厌恶", "害羞", "害怕",
    "生气", "认真", "紧张", "慌张",
    "疑惑", "兴奋", "无奈", "担心",
    "惊讶", "哭泣", "心动", "难为情", "自信", "调皮"
]
NUM_LABELS = len(TARGET_EMOTIONS) # 获取标签数量

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 模型和超参数预设 ---
PRESETS = {
    "BERT-Base": {
        "model_path": "bert-base-chinese",
        "description": "标准 BERT-Base 中文模型 (默认)",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "num_train_epochs": 10,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "early_stopping_patience": 4
    },
    "RoBERTa-WWM-Ext": {
        "model_path": "hfl/chinese-roberta-wwm-ext", 
        "description": "哈工大中文 RoBERTa-wwm-ext (通常性能更优)",
        "learning_rate": 1.5e-5,
        "per_device_train_batch_size": 16,
        "num_train_epochs": 10,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "early_stopping_patience": 4
    },
    "MacBERT-Base": {
        "model_path": "hfl/chinese-macbert-base",
        "description": "中文 MacBERT-Base (快速轻量级替代)", 
        "learning_rate": 3e-5, 
        "per_device_train_batch_size": 32, 
        "num_train_epochs": 30,
        "weight_decay": 0.05,
        "warmup_ratio": 0.15,
        "early_stopping_patience": 5
    }
}

# --- 交互式模型选择逻辑 ---

def get_config_interactive():
    """交互式地让用户选择模型配置"""
    print("\n=========================================")
    print("🤖 请选择要用于情绪分类的模型：")
    
    # 打印可选项
    keys = list(PRESETS.keys())
    default_key = "BERT-Base"
    
    for i, key in enumerate(keys):
        desc = PRESETS[key]["description"]
        # 移除描述中的 "(默认)" 标记，只在提示行使用
        display_desc = desc.replace(" (默认)", "")
        print(f"  [{i+1}] {key}: {display_desc}")

    print("-----------------------------------------")
    print(f"输入数字序号选择，或直接回车使用默认模型 [{default_key}]。")

    while True:
        try:
            # 兼容性处理，防止在某些环境中 input() 报错
            if not sys.stdin.isatty():
                print("\n检测到非交互式环境，使用默认模型: BERT-Base")
                selected_key = default_key
                break

            choice = input("您的选择: ").strip()
            
            if not choice:
                # 默认选择 BERT-Base
                selected_key = default_key
                break
            
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(keys):
                selected_key = keys[choice_index]
                break
            else:
                print("⚠️ 输入无效，请重新输入列表中的数字序号。")
        except ValueError:
            print("⚠️ 输入无效，请输入数字或直接回车。")
        except EOFError:
            print("\n检测到非交互式环境，使用默认模型: BERT-Base")
            selected_key = default_key
            break

    # 返回选中的配置
    return selected_key, PRESETS[selected_key]


# --- 配置加载和导出 ---
# 在模块导入时运行选择逻辑
MODEL_CHOICE, CURRENT_CONFIG = get_config_interactive()
MODEL_PATH = CURRENT_CONFIG["model_path"]

# 确保 train_emotion_classifier.py 可以导入这些设置
__all__ = [
    'DATA_PATH', 'CACHE_PATH', 'FINAL_MODEL_DIR', 'SEED', 
    'TARGET_EMOTIONS', 'NUM_LABELS', 'DEVICE', 
    'MODEL_PATH', 'CURRENT_CONFIG', 'MODEL_CHOICE', 'PRESETS'
]