#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
私募基金问答助手 - 全面测试套件
覆盖所有22个规则的多种测试问题
采用更接近真实用户提问的查询方式
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fund_qa_langgraph_v2 import FundQAAssistant

def find_rule_by_question(result, assistant):
    """
    从结果中找出匹配的规则ID
    :param result: 查询结果文本
    :param assistant: FundQAAssistant实例
    :return: 匹配的规则ID，未找到返回None
    """
    for rule in assistant.rules_db:
        if rule['question'] in result:
            return rule['id']
    return None

def main():
    """
    主测试函数，覆盖全部22条规则的多种提问方式
    """
    assistant = FundQAAssistant()

    # 测试用例 - 覆盖全部22条规则，每条规则2-4种提问方式
    test_cases = [
        # ========== rule001: 合格投资者标准 ==========
        {"query": "私募基金的合格投资者标准是什么？", "rule_id": "rule001", "keywords": ["合格投资者", "100万元"]},
        {"query": "什么是合格投资者", "rule_id": "rule001", "keywords": ["100万"]},
        {"query": "投资私募基金的最低金额要求", "rule_id": "rule001", "keywords": ["100万"]},
        {"query": "我想投资私募基金，需要什么条件", "rule_id": "rule001", "keywords": ["合格"]},

        # ========== rule002: 最低募集规模 ==========
        {"query": "私募基金的最低募集规模要求是多少？", "rule_id": "rule002", "keywords": ["1000万元"]},
        {"query": "成立私募基金需要多少钱", "rule_id": "rule002", "keywords": ["1000万"]},

        # ========== rule003: 管理人资质 ==========
        {"query": "私募基金管理人需要什么资质？", "rule_id": "rule003", "keywords": ["登记"]},
        {"query": "怎么成为私募基金管理人", "rule_id": "rule003", "keywords": ["协会"]},

        # ========== rule004: 募集期 ==========
        {"query": "私募基金募集期一般是多长时间？", "rule_id": "rule004", "keywords": ["6个月"]},
        {"query": "募集期多长时间", "rule_id": "rule004", "keywords": ["6个月"]},

        # ========== rule005: 风险准备金 ==========
        {"query": "私募基金管理人的风险准备金要求是什么？", "rule_id": "rule005", "keywords": ["10%"]},
        {"query": "风险准备金比例是多少", "rule_id": "rule005", "keywords": ["10%"]},

        # ========== rule006: 风险等级 ==========
        {"query": "私募基金的风险等级如何划分？", "rule_id": "rule006", "keywords": ["R1", "R5"]},
        {"query": "风险等级有哪些", "rule_id": "rule006", "keywords": ["谨慎型", "激进型"]},
        {"query": "R3是什么风险等级", "rule_id": "rule006", "keywords": ["平衡型"]},

        # ========== rule007: 管理人责任 ==========
        {"query": "私募基金管理人应当履行什么责任？", "rule_id": "rule007", "keywords": ["忠实义务"]},
        {"query": "管理人的义务有哪些", "rule_id": "rule007", "keywords": ["勤勉义务"]},

        # ========== rule008: 信息披露 ==========
        {"query": "私募基金需要向投资者披露哪些信息？", "rule_id": "rule008", "keywords": ["基金净值"]},
        {"query": "信息披露包括什么内容", "rule_id": "rule008", "keywords": ["风险提示"]},

        # ========== rule009: 基金合同 ==========
        {"query": "私募基金的基金合同必须包含什么内容？", "rule_id": "rule009", "keywords": ["权利义务"]},
        {"query": "合同要写明什么", "rule_id": "rule009", "keywords": ["收益分配"]},

        # ========== rule010: 监管部门报告 ==========
        {"query": "私募基金需要向监管部门报告什么信息？", "rule_id": "rule010", "keywords": ["定期报告"]},
        {"query": "向协会要报告什么信息", "rule_id": "rule010", "keywords": ["重大事项"]},

        # ========== rule011: 投资范围 ==========
        {"query": "私募基金可以投资哪些资产？", "rule_id": "rule011", "keywords": ["股票", "债券"]},
        {"query": "投资什么资产", "rule_id": "rule011", "keywords": ["衍生品"]},

        # ========== rule012: 投资集中度 ==========
        {"query": "私募基金投资集中度有什么限制？", "rule_id": "rule012", "keywords": ["20%"]},
        {"query": "集中度比例是多少", "rule_id": "rule012", "keywords": ["20%"]},

        # ========== rule013: 费用结构 ==========
        {"query": "私募基金的费用通常有哪些？", "rule_id": "rule013", "keywords": ["管理费", "业绩报酬"]},
        {"query": "有什么费用", "rule_id": "rule013", "keywords": ["保管费"]},

        # ========== rule014: 管理费 ==========
        {"query": "什么是管理费？如何计算？", "rule_id": "rule014", "keywords": ["管理费"]},
        {"query": "管理费怎么计算", "rule_id": "rule014", "keywords": ["年度百分比"]},

        # ========== rule015: 业绩报酬 ==========
        {"query": "什么是业绩报酬？计提条件是什么？", "rule_id": "rule015", "keywords": ["超额收益"]},
        {"query": "业绩报酬怎么算", "rule_id": "rule015", "keywords": ["正收益"]},

        # ========== rule016: 退出机制 ==========
        {"query": "投资者如何从私募基金中退出？", "rule_id": "rule016", "keywords": ["退出"]},
        {"query": "怎么退出私募基金", "rule_id": "rule016", "keywords": ["赎回"]},

        # ========== rule017: 清算分配 ==========
        {"query": "私募基金清算时应该如何分配资产？", "rule_id": "rule017", "keywords": ["清算"]},
        {"query": "清算如何分配", "rule_id": "rule017", "keywords": ["清算费用"]},

        # ========== rule018: 强制清算 ==========
        {"query": "什么情况下私募基金会被强制清算？", "rule_id": "rule018", "keywords": ["强制清算"]},
        {"query": "什么情况基金要清算", "rule_id": "rule018", "keywords": ["撤销牌照"]},

        # ========== rule019: 主要风险 ==========
        {"query": "私募基金的主要风险有哪些？", "rule_id": "rule019", "keywords": ["市场风险"]},
        {"query": "有什么风险", "rule_id": "rule019", "keywords": ["流动性风险"]},

        # ========== rule020: 风险管理 ==========
        {"query": "私募基金管理人应该如何进行风险管理？", "rule_id": "rule020", "keywords": ["风险管理"]},
        {"query": "怎样进行风险管理", "rule_id": "rule020", "keywords": ["压力测试"]},

        # ========== rule021: 合规要求 ==========
        {"query": "私募基金管理人需要符合什么合规要求？", "rule_id": "rule021", "keywords": ["合规"]},
        {"query": "有什么合规义务", "rule_id": "rule021", "keywords": ["利益冲突"]},

        # ========== rule022: 禁止行为 ==========
        {"query": "私募基金不能做的事情有什么？", "rule_id": "rule022", "keywords": ["禁止"]},
        {"query": "什么是违法违规", "rule_id": "rule022", "keywords": ["内幕交易"]},
    ]

    print("\n" + "=" * 70)
    print("私募基金问答助手 - 全面测试套件")
    print(f"共 {len(test_cases)} 个测试用例，覆盖全部22条规则")
    print("=" * 70 + "\n")

    passed = 0
    failed = 0
    rule_coverage = set()

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected_rule = test_case["rule_id"]
        keywords = test_case["keywords"]

        print(f"测试 {i}/{len(test_cases)}: {query}")
        result = assistant.process_query(query)

        # 检查关键词是否在答案中
        keywords_found = all(kw in result for kw in keywords)

        # 获取匹配的规则ID
        matched_rule = find_rule_by_question(result, assistant)

        # 获取答案的第一行（标题）
        first_line = result.split('\n')[0]

        if keywords_found:
            print(f"  [PASS]")
            print(f"    {first_line[:60]}")
            passed += 1
            rule_coverage.add(expected_rule)
        else:
            print(f"  [FAIL]")
            missing_kw = [kw for kw in keywords if kw not in result]
            print(f"    缺少关键词：{missing_kw}")
            print(f"    返回：{first_line[:60]}...")
            failed += 1
            rule_coverage.add(expected_rule)

        print()

    print("=" * 70)
    print(f"测试结果：{passed} 通过，{failed} 失败，共 {len(test_cases)} 个测试")
    print(f"规则覆盖率：{len(rule_coverage)}/22 条规则被测试")
    if len(rule_coverage) < 22:
        uncovered = set(f"rule{str(i).zfill(3)}" for i in range(1, 23)) - rule_coverage
        print(f"未覆盖规则：{uncovered}")
    print("=" * 70 + "\n")

    return passed, failed

if __name__ == "__main__":
    main()
