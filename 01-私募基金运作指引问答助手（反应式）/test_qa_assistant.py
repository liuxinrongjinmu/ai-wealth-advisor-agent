#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
私募基金问答助手测试脚本
"""

import sys
sys.path.insert(0, r'C:\Users\10300\Desktop\my-project\智能投顾AI助手系统\01-私募基金运作指引问答助手（反应式）')

from fund_qa_langgraph_v2 import FundQAAssistant

def main():
    """主测试函数"""
    assistant = FundQAAssistant()
    
    # 测试问题集合 - 覆盖各个主要功能
    test_cases = [
        {
            "query": "私募基金的合格投资者标准是什么？",
            "category": "设立与募集",
            "expected_keywords": ["合格投资者", "100万元"]
        },
        {
            "query": "私募基金可以投资哪些资产？",
            "category": "投资范围",
            "expected_keywords": ["股票", "债券", "衍生品"]
        },
        {
            "query": "风险等级有哪些",
            "category": "监管规定",
            "expected_keywords": ["R1", "R2", "R3", "R4", "R5"]
        },
        {
            "query": "私募基金有哪些费用",
            "category": "费用结构",
            "expected_keywords": ["管理费", "业绩报酬"]
        },
        {
            "query": "风险",
            "category": "风险管理",
            "expected_keywords": ["风险"]
        },
        {
            "query": "私募基金有哪些风险",
            "category": "风险管理",
            "expected_keywords": ["市场风险", "流动性风险"]
        },
        {
            "query": "私募基金有什么禁止行为",
            "category": "合规要求",
            "expected_keywords": ["禁止", "不能"]
        }
    ]
    
    print("\n" + "="*70)
    print("私募基金问答助手完整测试")
    print("="*70 + "\n")
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        category = test_case["category"]
        keywords = test_case["expected_keywords"]
        
        print(f"测试 {i}: {query}")
        result = assistant.process_query(query)
        
        # 检查分类是否正确
        category_match = category in result
        
        # 检查关键字是否在答案中
        keywords_found = all(kw in result for kw in keywords)
        
        # 获取答案的第一行（标题）
        first_line = result.split('\n')[0]
        
        if category_match and keywords_found:
            print(f"  [PASS]")
            print(f"    {first_line}")
            passed += 1
        else:
            print(f"  [FAIL]")
            if not category_match:
                print(f"    分类错误：找不到'{category}'")
            if not keywords_found:
                missing_kw = [kw for kw in keywords if kw not in result]
                print(f"    缺少关键词：{missing_kw}")
            print(f"    返回：{first_line[:60]}...")
            failed += 1
        
        print()
    
    print("="*70)
    print(f"测试结果：{passed} 通过，{failed} 失败，共 {len(test_cases)} 个测试")
    print("="*70 + "\n")
    
    return passed, failed

if __name__ == "__main__":
    main()
