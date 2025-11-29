import torch
import pandas as pd
import numpy as np
import os
import shutil
import json
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

# --- 导入配置和工具 ---
# 当导入 config 时，模型选择的交互式流程已经运行完毕，变量已被设置
from config import (
    DATA_PATH, CACHE_PATH, FINAL_MODEL_DIR, SEED, 
    TARGET_EMOTIONS, NUM_LABELS, DEVICE, 
    MODEL_PATH, CURRENT_CONFIG, MODEL_CHOICE
)
# 假设 cleancache.py 包含 clear_folder 函数
from cleancache import clear_folder

# 打印当前使用的配置 (现在这些变量是基于用户选择或默认值设置的)
print("\n=========================================")
# 移除描述中的默认标记，让输出更整洁
display_desc = CURRENT_CONFIG['description'].replace(" (默认)", "")
print(f"🚀 模型选择: {MODEL_CHOICE} ({display_desc})")
print(f"🧠 模型路径: {MODEL_PATH}")
print(f"💻 运行设备: {DEVICE}")
print(f"🎯 标签数量: {NUM_LABELS} (情绪类别)")
print(f"📊 核心超参数:")
for k, v in CURRENT_CONFIG.items():
    if k not in ["model_path", "description"]:
        print(f"   - {k}: {v}")
print("=========================================")

# 固定随机种子保证可复现
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_data(data_path=DATA_PATH):
    """加载并预处理数据，强制使用定义的情绪标签"""
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"错误：数据文件 '{data_path}' 未找到。请确保文件存在于正确路径。")
        # 退出程序
        exit()
    
    # 筛选有效标签，并转换为字符串以防万一
    data["label"] = data["label"].astype(str)
    # 强制只保留 TARGET_EMOTIONS 中的标签
    data = data[data["label"].isin(TARGET_EMOTIONS)].copy() 
    data["text"] = data["text"].astype(str)

    if data.empty:
        print(f"错误：在 '{data_path}' 中没有找到属于 TARGET_EMOTIONS 的数据。")
        exit()

    # 数据统计
    print("\n=== 数据统计 ===")
    print("筛选后总样本数:", len(data))
    
    # 使用固定顺序的标签编码器
    label_encoder = LabelEncoder()
    label_encoder.fit(TARGET_EMOTIONS)  # 强制按定义顺序编码

    # 划分数据集：保证测试集至少包含每个类别一个样本，并尝试分层
    test_size = 0.2 # 默认测试集比例
    if len(data) < NUM_LABELS * 2:
         print("警告：数据量过少，分层抽样可能失败或效果不佳。")
         test_size = 0.1

    try:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            data["text"].tolist(),
            data["label"].tolist(),
            test_size=test_size,
            stratify=data["label"], # 尝试分层抽样
            random_state=SEED
        )
    except ValueError as e:
        print(f"分层抽样失败: {e}. 尝试非分层抽样...")
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
    print("标签映射:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    return train_texts, test_texts, train_labels_encoded, test_labels_encoded, label_encoder


class EmotionDataset(torch.utils.data.Dataset):
    """自定义数据集类，用于Trainer"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.as_tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.as_tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_and_evaluate():
    """训练和评估情绪分类模型的主流程""" 
    # 1. 加载数据
    train_texts, test_texts, train_labels, test_labels, label_encoder = load_data()

    # 2. 初始化模型和分词器
    print("\n初始化分词器和模型...")
    # 使用 use_fast=False 避免某些特殊模型的问题，但通常 fast=True 更快
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, use_fast=False) 
    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=NUM_LABELS,
        # 传递标签映射给模型配置，便于部署
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)}
    )

    # 3. 数据编码
    print("\nTokenizing 数据...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    # 4. 创建数据集
    train_dataset = EmotionDataset(train_encodings, train_labels)
    test_dataset = EmotionDataset(test_encodings, test_labels)

    # 5. 训练配置（使用 config.py 中的超参数）
    training_args = TrainingArguments(
        output_dir=CACHE_PATH,                       # 临时检查点和日志目录
        num_train_epochs=CURRENT_CONFIG["num_train_epochs"],
        per_device_train_batch_size=CURRENT_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=32,
        learning_rate=CURRENT_CONFIG["learning_rate"],
        weight_decay=CURRENT_CONFIG["weight_decay"],
        warmup_ratio=CURRENT_CONFIG["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",         # 按加权 F1 选择最佳模型
        greater_is_better=True,
        logging_dir=f'{CACHE_PATH}/logs',            # 指定日志目录
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),              # 如果可用，自动启用混合精度
        report_to="none"                             # 禁用外部报告
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
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=CURRENT_CONFIG["early_stopping_patience"] # 从配置中获取早停耐心值
        )]
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
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    print(f"\n保存最佳模型到 {FINAL_MODEL_DIR}...")
    trainer.save_model(FINAL_MODEL_DIR) # 保存最佳模型、tokenizer配置、训练状态等
    tokenizer.save_pretrained(FINAL_MODEL_DIR) # 确保 tokenizer 也保存

    # 保存标签映射
    label_mapping_path = os.path.join(FINAL_MODEL_DIR, "label_mapping.json")
    print(f"保存标签映射到 {label_mapping_path}...")
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            "id2label": {str(i): label for i, label in enumerate(label_encoder.classes_)},
            "label2id": {label: i for i, label in enumerate(label_encoder.classes_)}
        }, f, ensure_ascii=False, indent=2)

    print(f"\n模型和配置已保存到 {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    train_and_evaluate()
    print("是否清空训练缓存文件夹 results_18emo？(y/n)")
    # 使用 input() 来获取用户输入
    user_input = input().lower().strip() 
    if user_input == "y":
        clear_folder(CACHE_PATH)
        print("缓存已清空")
    else:
        print(f"缓存未清空，保存在:{CACHE_PATH}")
    print("训练完成！")