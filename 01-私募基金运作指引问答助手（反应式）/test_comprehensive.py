#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
私募基金问答助手 - 全面测试套件 (改进版)
覆盖所有22个规则的多种测试问题
采用更接近真实用户提问的查询方式
"""

import sys
sys.path.insert(0, r'C:\Users\10300\Desktop\my-project\智能投顾AI助手系统\01-私募基金运作指引问答助手（反应式）')

from fund_qa_langgraph_v2 import FundQAAssistant

def find_rule_by_question(result, assistant):
    """从结果中找出匹配的规则ID"""
    for rule in assistant.rules_db:
        if rule['question'] in result:
            return rule['id']
    return None

def main():
    """主测试函数"""
    assistant = FundQAAssistant()
    
    # 改进的测试用例 - 更贴近真实用户提问
    test_cases = [
        # ========== rule001: 合格投资者标准 ========== 
        {"query": "私募基金的合格投资者标准是什么？", "rule_id": "rule001", "keywords": ["合格投资者", "100万元"]},
        {"query": "什么是合格投资者", "rule_id": "rule001", "keywords": ["100万"]},
        {"query": "投资私募基金的最低金额要求", "rule_id": "rule001", "keywords": ["100万"]},
        {"query": "我想投资私募基金，需要什么条件", "rule_id": "rule001", "keywords": ["合格"]},
        # ...existing code...
    ]
    # ...existing code...

if __name__ == "__main__":
    main()
