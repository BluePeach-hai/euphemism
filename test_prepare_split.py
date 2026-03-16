#!/usr/bin/env python

import os
import os.path as osp
import json

from euphemism.data import EuphemismDataModule

# 创建一个简化版的数据模块，仅用于测试中文分词功能，避免网络连接
class SimpleDataModule:
    def __init__(self):
        import jieba
        self.jieba = jieba
    
    def tokenize_chinese(self, text):
        return ' '.join(self.jieba.cut(text))
    
    def prepare_split(self, split='dataset_text', file_path=None):
        # 读取中文数据集
        import csv
        import tqdm
        root = 'data'
        input_file = osp.join(root, f'{split}.csv')
        data = []
        
        print(f"读取中文数据集: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(tqdm.tqdm(reader, desc=f"处理 {split} 数据集")):
                # 提取CSV行数据
                text = row.get('text', '')
                is_drug_related = int(row.get('is_drug_related', 0))
                original_keyword = row.get('原始关键词', '')
                final_keyword = row.get('最终合并关键词', '')
                keywords = row.get('keywords', '')
                main_type = row.get('main_type', '')
                
                # 进行中文分词
                segmented_text = self.tokenize_chinese(text)
                segmented_keywords = self.tokenize_chinese(keywords) if keywords else ''
                
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
        
        if file_path:
            # 保存JSON文件，确保使用正确的编码
            os.makedirs(osp.dirname(file_path), exist_ok=True)
            with open(osp.abspath(file_path), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"成功保存分词结果到: {file_path}")
        
        return data

# 使用简化版数据模块测试中文分词功能
print("创建简化版数据模块实例...")
dm = SimpleDataModule()

# 直接调用prepare_split方法处理中文数据集
print("处理中文数据集...")
dm.prepare_split(split='dataset_text', file_path='data/dataset_text_test.json')

# 检查是否生成了JSON文件
if osp.exists('data/dataset_text_test.json'):
    print("成功生成中文分词数据集JSON文件")
    
    # 查看生成的JSON文件内容
    with open('data/dataset_text_test.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"生成了 {len(data)} 条数据")
    
    # 查看前几条数据
    print("\n查看前2条数据:")
    for i in range(min(2, len(data))):
        item = data[i]
        print(f"\n数据 {i+1}:")
        print(f"原文: {item['text']}")
        print(f"分词后: {item['segmented_text']}")
        print(f"是否涉毒: {item['is_drug_related']}")
        print(f"关键词: {item['keywords']}")
        print(f"分词关键词: {item['segmented_keywords']}")
        print(f"主要类型: {item['main_type']}")
else:
    print("生成中文分词数据集JSON文件失败")

print("\n测试完成！")
