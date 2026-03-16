#!/usr/bin/env python

import json
import os
import random
from sklearn.model_selection import train_test_split
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 读取数据集
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"成功加载数据集，总共有 {len(data)} 条数据")
    return data

# 分析数据集
def analyze_dataset(data):
    total = len(data)
    drug_related = sum(1 for x in data if x["is_drug_related"] == 1)
    non_drug_related = sum(1 for x in data if x["is_drug_related"] == 0)
    
    logging.info(f"涉毒数据: {drug_related} ({drug_related/total*100:.2f}%)")
    logging.info(f"非涉毒数据: {non_drug_related} ({non_drug_related/total*100:.2f}%)")
    
    # 检查数据分布
    main_types = {}
    for item in data:
        main_type = item["main_type"]
        main_types[main_type] = main_types.get(main_type, 0) + 1
    
    logging.info("数据类型分布:")
    for main_type, count in main_types.items():
        logging.info(f"  {main_type}: {count} ({count/total*100:.2f}%)")

# 划分数据集
def split_data(data, test_size=0.2, random_state=42):
    # 按类别分层采样
    drug_related_data = [x for x in data if x["is_drug_related"] == 1]
    non_drug_related_data = [x for x in data if x["is_drug_related"] == 0]
    
    # 对两类数据分别进行划分
    drug_related_val, drug_related_test = train_test_split(
        drug_related_data, test_size=test_size, random_state=random_state
    )
    
    non_drug_related_val, non_drug_related_test = train_test_split(
        non_drug_related_data, test_size=test_size, random_state=random_state
    )
    
    # 合并划分后的数据
    val_data = drug_related_val + non_drug_related_val
    test_data = drug_related_test + non_drug_related_test
    
    # 打乱顺序
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    # 更新索引
    for i, item in enumerate(val_data):
        item["index"] = i
    
    for i, item in enumerate(test_data):
        item["index"] = i
    
    logging.info(f"验证集大小: {len(val_data)} (涉毒: {len(drug_related_val)}, 非涉毒: {len(non_drug_related_val)})")
    logging.info(f"测试集大小: {len(test_data)} (涉毒: {len(drug_related_test)}, 非涉毒: {len(non_drug_related_test)})")
    
    return val_data, test_data

# 保存数据集
def save_dataset(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"数据集已保存到: {file_path}")

# 主函数
def main():
    # 配置
    input_file = 'data/dataset_text.json'
    val_output_file = 'data/dataset_text_val.json'
    test_output_file = 'data/dataset_text_test.json'
    test_size = 0.5  # 验证集和测试集各占50%
    random_state = 42
    
    # 加载数据集
    data = load_dataset(input_file)
    
    # 分析数据集
    analyze_dataset(data)
    
    # 划分数据集
    val_data, test_data = split_data(data, test_size=test_size, random_state=random_state)
    
    # 保存数据集
    save_dataset(val_data, val_output_file)
    save_dataset(test_data, test_output_file)
    
    logging.info("数据集划分完成!")

if __name__ == "__main__":
    main()