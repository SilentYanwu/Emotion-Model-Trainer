import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import os
import shutil
import json

from cleancache import clear_folder
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 设置模型和文件路径
MODEL_PATH = "bert-base-chinese"
DATA_PATH = "emotion_data_manual.csv"
CACHE_PATH = "results_18emo"

# 固定随机种子保证可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 明确定义18类情绪及其顺序（关键！）
TARGET_EMOTIONS = ["高兴", "厌恶", "害羞", "害怕",
                   "生气", "认真", "紧张", "慌张",
                   "疑惑", "兴奋", "无奈", "担心",
                   "惊讶", "哭泣", "心动", "难为情", "自信", "调皮"]
NUM_LABELS = len(TARGET_EMOTIONS) # 获取标签数量

def load_data(data_path=DATA_PATH):
    """加载并预处理数据，强制使用定义的情绪标签"""
    # 加载数据并筛选目标情绪
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"错误：数据文件 '{data_path}' 未找到。请确保文件存在于正确路径。")
        exit() # 或者引发异常
    
    # 筛选有效标签，并转换为字符串以防万一
    data["label"] = data["label"].astype(str)
    data = data[data["label"].isin(TARGET_EMOTIONS)].copy() # 使用 .copy() 避免 SettingWithCopyWarning
    data["text"] = data["text"].astype(str)

    if data.empty:
        print(f"错误：在 '{data_path}' 中没有找到属于 TARGET_EMOTIONS 的数据。")
        exit()

    # 数据统计
    print("\n=== 数据统计 ===")
    print(f"目标情绪类别数量: {NUM_LABELS}")
    print("筛选后总样本数:", len(data))
    print("类别分布:\n", data["label"].value_counts())

    # 使用固定顺序的标签编码器
    label_encoder = LabelEncoder()
    label_encoder.fit(TARGET_EMOTIONS)  # 强制按定义顺序编码

    # 划分数据集（保证测试集至少包含每个类别一个样本，如果可能）
    # 计算最小测试集比例以包含所有类
    min_samples_per_class = 1
    required_test_samples = NUM_LABELS * min_samples_per_class
    min_test_size_for_all_classes = required_test_samples / len(data)

    # 设置测试集比例，通常在0.1到0.3之间，但要确保能覆盖所有类
    test_size = max(0.2, min(min_test_size_for_all_classes, 0.3))
    # 如果总样本太少，可能无法满足 stratify 要求，这里简化处理
    if len(data) < NUM_LABELS * 2: # 至少保证训练集和测试集每个类都有样本（理论上）
         print("警告：数据量过少，可能无法有效分层或训练。")
         test_size = max(0.1, min_test_size_for_all_classes) # 尝试减少测试集比例

    print(f"实际使用的测试集比例: {test_size:.2f}")

    try:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            data["text"].tolist(),
            data["label"].tolist(),
            test_size=test_size,
            stratify=data["label"], # 尝试分层抽样
            random_state=SEED
        )
    except ValueError as e:
        print(f"分层抽样失败: {e}. 可能某些类别样本过少。尝试非分层抽样...")
        # 如果分层失败（通常因为某类样本太少），退回到普通随机抽样
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            data["text"].tolist(),
            data["label"].tolist(),
            test_size=test_size,
            random_state=SEED
        )

    # 编码标签
    train_labels_encoded = label_encoder.transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    print(f"\n划分结果: 训练集={len(train_texts)}, 测试集={len(test_texts)}")
    print("测试集类别分布:\n", pd.Series(test_labels).value_counts().sort_index())
    # 检查测试集是否包含所有类别
    test_unique_labels = set(test_labels)
    if len(test_unique_labels) < NUM_LABELS:
        print(f"警告：测试集仅包含 {len(test_unique_labels)}/{NUM_LABELS} 个类别。缺失的类别：{set(TARGET_EMOTIONS) - test_unique_labels}")

    print("标签映射:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    return train_texts, test_texts, train_labels_encoded, test_labels_encoded, label_encoder

class EmotionDataset(torch.utils.data.Dataset):
    """自定义数据集类"""
    # *** FIX: Renamed 'init' to '__init__' ***
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Ensure encodings contain torch tensors or convert numpy arrays
        item = {key: torch.as_tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.as_tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_and_evaluate():
    """训练和评估18类情绪分类模型""" # <-- 更新注释

    # 1. 加载数据
    train_texts, test_texts, train_labels, test_labels, label_encoder = load_data()

    # 2. 初始化模型和分词器
    model_name = MODEL_PATH
    tokenizer = BertTokenizer.from_pretrained(model_name, use_fast=False)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,  # 使用变量 NUM_LABELS
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)}
    )

    # 3. 数据编码
    print("\nTokenizing 数据...")
    # 使用 tolist() 确保输入是 list of strings
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        # return_tensors="pt" # Trainer 会处理 tensor 转换, 这里可以不指定或者指定 None
    )
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        # return_tensors="pt"
    )

    # 4. 创建数据集
    train_dataset = EmotionDataset(train_encodings, train_labels)
    test_dataset = EmotionDataset(test_encodings, test_labels)

    # 5. 训练配置（优化后的超参数）
    # *** FIX: Updated output_dir name ***
    output_dir_base = "./results_18emo"
    training_args = TrainingArguments(
        output_dir=output_dir_base,        # 输出目录
        num_train_epochs=10,               # 根据需要调整 epoch
        per_device_train_batch_size=16,    # 根据显存调整 batch size : 16 32 12 8 等等，我个人实践下来，12/16差不多最佳，但是速度偏慢。
        # 实践经验：可以提高批次（16到32）来提高运行速度，但同时建议提高学习率(2e-5到4e-5），正则化（从0.01到0.1）和预热比例（从0.1到0.3）。
        per_device_eval_batch_size=32,
        learning_rate=2e-5,               # 适合 fine-tuning 的学习率
        weight_decay=0.01,                 # L2 正则化
        warmup_ratio=0.1,                  # 预热比例
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # save_strategy="no",
        # load_best_model_at_end=False,
        metric_for_best_model="f1_weighted", # 按加权 F1 选择最佳模型
        greater_is_better=True,
        logging_dir=f'{output_dir_base}/logs', # 指定日志目录
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),    # 如果可用，自动启用混合精度
        report_to="none"                   # 禁用 wandb 等外部报告
    )

    # 6. 自定义评估指标
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        # 使用 label_encoder.classes_ 获取正确的标签名称顺序
        report = classification_report(
            labels, preds,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0 # 处理某个类别在预测或真实标签中都没有出现的情况
        )
        # 返回 Trainer 需要的指标
        return {
            "accuracy": report["accuracy"],
            "f1_weighted": report["weighted avg"]["f1-score"],
            "precision_weighted": report["weighted avg"]["precision"],
            "recall_weighted": report["weighted avg"]["recall"],
        }

    # 7. 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer, # 传递 tokenizer 方便 Trainer 处理 padding
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]    # 早停下
    )

    print("\n开始训练...")
    trainer.train()

    # 8. 最终评估 (加载最好的模型进行评估)
    print("\n=== 测试集最终性能 (使用最佳模型) ===")
    eval_results = trainer.evaluate(test_dataset)
    print(f"评估结果: {eval_results}")

    # 获取详细的分类报告
    print("\n详细分类报告:")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    print(classification_report(
        test_labels, # 使用原始编码的 test_labels
        y_pred,
        target_names=label_encoder.classes_, # 使用正确的标签名称
        digits=4
    ))

    # 9. 保存模型和配置
    # *** FIX: Updated output_dir name ***
    final_model_dir = "./emotion_model_18emo"
    os.makedirs(final_model_dir, exist_ok=True)
    print(f"\n保存最佳模型到 {final_model_dir}...")
    trainer.save_model(final_model_dir) # 保存最佳模型、tokenizer配置、训练状态等
    tokenizer.save_pretrained(final_model_dir) # 确保 tokenizer 也保存

    # 保存标签映射
    label_mapping_path = os.path.join(final_model_dir, "label_mapping.json")
    print(f"保存标签映射到 {label_mapping_path}...")
    # *** FIX: Added encoding='utf-8' to open() ***
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            # 确保 key 是字符串，因为 JSON 的 key 必须是 string
            "id2label": {str(i): label for i, label in enumerate(label_encoder.classes_)},
            "label2id": {label: i for i, label in enumerate(label_encoder.classes_)}
        }, f, ensure_ascii=False, indent=2)

    print(f"\n模型和配置已保存到 {final_model_dir}")

if __name__ == "__main__":
    train_and_evaluate()
    print("是否清空缓存？(y/n)")
    if input().lower() == "y":
        clear_folder(CACHE_PATH)
        print("缓存已清空")
    else:
        print("缓存未清空，保存在:"+CACHE_PATH)
    print("训练完成！")

