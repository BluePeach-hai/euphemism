#!/usr/bin/env python

import sys
import os.path as osp
import json
import numpy as np
from transformers import AutoTokenizer

# 添加项目根目录到Python路径
project_root = osp.abspath(osp.join(__file__, '../..'))
sys.path.insert(0, project_root)

from euphemism.data import _helper

# 加载数据集
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 分析序列长度
def analyze_sequence_lengths(dataset, text_field='segmented_text', tokenizer=None):
    lengths = []
    for item in dataset:
        text = item[text_field]
        if tokenizer:
            # 使用tokenizer计算token数量
            tokens = tokenizer.tokenize(text)
            length = len(tokens)
        else:
            # 简单计算字符数量
            length = len(text)
        lengths.append(length)
    
    print(f"已处理 {len(lengths)} 个样本")
    
    # 计算统计信息
    lengths = np.array(lengths)
    print(f"总样本数: {len(lengths)}")
    print(f"最小长度: {np.min(lengths)}")
    print(f"最大长度: {np.max(lengths)}")
    print(f"平均长度: {np.mean(lengths):.2f}")
    print(f"中位数长度: {np.median(lengths):.2f}")
    print(f"90%分位数: {np.percentile(lengths, 90):.2f}")
    print(f"95%分位数: {np.percentile(lengths, 95):.2f}")
    print(f"99%分位数: {np.percentile(lengths, 99):.2f}")
    
    # 统计超过不同阈值的样本比例
    thresholds = [128, 256, 384, 512]
    for threshold in thresholds:
        ratio = (lengths > threshold).mean() * 100
        print(f"超过 {threshold} 长度的样本比例: {ratio:.2f}%")
    
    return lengths

if __name__ == "__main__":
    # 加载中文数据集
    data_path = osp.join(project_root, 'data', 'dataset_text.json')
    if not osp.exists(data_path):
        print(f"数据集文件不存在: {data_path}")
        print("请先运行数据准备脚本")
        sys.exit(1)
    
    dataset = load_dataset(data_path)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base', use_fast=True)
    
    print("\n=== 序列长度分析 (字符数) ===")
    analyze_sequence_lengths(dataset, text_field='text')
    
    print("\n=== 序列长度分析 (分词后) ===")
    analyze_sequence_lengths(dataset, text_field='segmented_text', tokenizer=tokenizer)
    
    print("\n=== 序列长度分析 (Token数) ===")
    analyze_sequence_lengths(dataset, text_field='segmented_text', tokenizer=tokenizer)
