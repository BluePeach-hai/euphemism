#!/usr/bin/env python

import os
import os.path as osp
import json
import csv
import jieba
from tqdm import tqdm

# 中文分词函数
def tokenize_chinese(text):
    """对中文文本进行分词"""
    return ' '.join(jieba.cut(text))

# 处理中文数据集函数
def process_chinese_dataset(input_file, output_file):
    """处理中文数据集并生成分词结果"""
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(tqdm(reader, desc="处理中文数据集")):
            # 提取CSV行数据
            text = row.get('text', '')
            is_drug_related = int(row.get('is_drug_related', 0))
            original_keyword = row.get('原始关键词', '')
            final_keyword = row.get('最终合并关键词', '')
            keywords = row.get('keywords', '')
            main_type = row.get('main_type', '')
            
            # 进行中文分词
            segmented_text = tokenize_chinese(text)
            segmented_keywords = tokenize_chinese(keywords) if keywords else ''
            
            # 保存处理后的数据
            data.append({
                'index': i,
                'text': text,
                'segmented_text': segmented_text,
                'is_drug_related': is_drug_related,
                'original_keyword': original_keyword,
                'final_keyword': final_keyword,
                'keywords': keywords,
                'segmented_keywords': segmented_keywords,
                'main_type': main_type,
            })
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return data

if __name__ == "__main__":
    # 定义文件路径
    input_file = 'data/dataset_text.csv'
    output_file = 'data/dataset_text_tokenized.json'
    
    print("开始处理中文数据集...")
    
    # 检查输入文件是否存在
    if not osp.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
    else:
        # 处理数据集
        data = process_chinese_dataset(input_file, output_file)
        
        print(f"\n成功处理完成！")
        print(f"输入文件：{input_file}")
        print(f"输出文件：{output_file}")
        print(f"处理数据条数：{len(data)}")
        
        # 显示前几条处理结果
        if data:
            print("\n前2条处理结果：")
            for i in range(min(2, len(data))):
                item = data[i]
                print(f"\n数据 {i+1}:")
                print(f"原文: {item['text']}")
                print(f"分词后: {item['segmented_text']}")
                print(f"是否涉毒: {item['is_drug_related']}")
                print(f"关键词: {item['keywords']}")
                print(f"分词关键词: {item['segmented_keywords']}")
                print(f"主要类型: {item['main_type']}")
