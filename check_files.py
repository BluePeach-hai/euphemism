#!/usr/bin/env python

"""
简单检查脚本，只验证数据集文件是否存在，不加载模型
"""

import os
import json

def check_file(file_path):
    """检查文件是否存在并打印信息"""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ {file_path} 存在")
            print(f"   数据条目数: {len(data)}")
            print(f"   数据字段: {list(data[0].keys())}")
            return True
        except Exception as e:
            print(f"✗ {file_path} 存在但无法读取: {e}")
            return False
    else:
        print(f"✗ {file_path} 不存在")
        return False

def main():
    print("=== 检查数据集文件 ===")
    
    files = [
        'data/dataset_text.json',
        'data/dataset_text_train.json',
        'data/dataset_text_test.json'
    ]
    
    all_exist = True
    for file in files:
        print(f"\n检查 {file}:")
        if not check_file(file):
            all_exist = False
    
    print(f"\n=== 总结 ===")
    if all_exist:
        print("✓ 所有数据集文件都已成功生成并可以正常读取！")
        return 0
    else:
        print("✗ 部分文件不存在或无法读取")
        return 1

if __name__ == "__main__":
    main()