#!/usr/bin/env python

"""
根据terms.tsv和describe.tsv生成包含详细描述的新文件
"""

import os
import os.path as osp


def map_term_descriptions():
    # 定义文件路径
    ROOT = osp.abspath(osp.join(__file__, '../..'))
    DATA_ROOT = osp.join(ROOT, 'data')
    
    terms_file = osp.join(DATA_ROOT, 'terms.tsv')
    describe_file = osp.join(DATA_ROOT, 'describe_cn.tsv')
    output_file = osp.join(DATA_ROOT, 'terms_descriptions_cn.tsv')
    
    # 读取describe.tsv，创建类型到描述的映射字典
    print(f"读取描述文件: {describe_file}")
    type_to_description = {}
    
    with open(describe_file, 'r', encoding='utf-8') as f:
        # 跳过表头
        next(f)
        for line in f:
            line = line.strip()
            if line:
                # 处理可能的多个制表符分隔
                parts = line.split('\t')
                # 第一个部分是类型定义
                term_type = parts[0].strip()
                # 剩余部分合并为描述
                description = '\t'.join(parts[1:]).strip()
                type_to_description[term_type] = description
    
    print(f"成功加载 {len(type_to_description)} 种类型的描述")
    
    # 读取terms.tsv并生成新文件
    print(f"读取术语文件: {terms_file}")
    print(f"生成新文件: {output_file}")
    
    with open(terms_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        # 读取表头
        header = f_in.readline().strip()
        # 写入新表头
        f_out.write(f"{header}\tdescription\n")
        
        # 处理每行术语
        term_count = 0
        matched_count = 0
        
        for line in f_in:
            line = line.strip()
            if line:
                term_count += 1
                # 拆分术语和类型
                parts = line.split('\t')
                if len(parts) >= 2:
                    term = parts[0].strip()
                    term_type = parts[1].strip()
                    
                    # 查找对应的描述
                    description = type_to_description.get(term_type, "无可用描述")
                    if description != "无可用描述":
                        matched_count += 1
                    
                    # 写入新行
                    f_out.write(f"{term}\t{term_type}\t{description}\n")
                else:
                    # 处理格式不正确的行
                    f_out.write(f"{line}\t无可用描述\n")
    
    print(f"处理完成！共 {term_count} 个术语，其中 {matched_count} 个找到匹配的描述")
    print(f"新文件已保存到: {output_file}")


if __name__ == "__main__":
    map_term_descriptions()
