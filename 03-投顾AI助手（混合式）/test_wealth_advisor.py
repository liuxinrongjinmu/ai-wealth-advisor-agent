#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
投顾AI助手（混合式）- 测试脚本

验证混合智能体的工作流创建、条件分支路由和响应生成
"""

import os
import sys

# 添加模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_wealth_advisor_langgraph import (
    create_wealth_advisor_workflow,
    SAMPLE_CUSTOMER_PROFILES
)


def test_workflow_creation():
    """
    测试工作流图创建是否成功
    """
    print("测试1: 工作流图创建")
    try:
        workflow = create_wealth_advisor_workflow()
        assert workflow is not None, "工作流创建失败"
        print("  [PASS] 工作流图创建成功")
        return True
    except Exception as e:
        print(f"  [FAIL] 工作流图创建失败: {str(e)}")
        return False


def test_customer_profiles():
    """
    测试客户画像数据完整性
    """
    print("\n测试2: 客户画像数据")
    required_fields = [
        "customer_id", "risk_tolerance", "investment_horizon",
        "financial_goals", "investment_preferences",
        "portfolio_value", "current_allocations"
    ]
    all_passed = True
    for cid, profile in SAMPLE_CUSTOMER_PROFILES.items():
        for field in required_fields:
            if field not in profile:
                print(f"  [FAIL] 客户 {cid} 缺少字段: {field}")
                all_passed = False
        # 检查资产配置比例之和是否约为1
        total_alloc = sum(profile.get("current_allocations", {}).values())
        if abs(total_alloc - 1.0) > 0.01:
            print(f"  [FAIL] 客户 {cid} 资产配置比例之和为 {total_alloc:.2f}，不为1")
            all_passed = False

    if all_passed:
        print(f"  [PASS] {len(SAMPLE_CUSTOMER_PROFILES)} 个客户画像数据完整")
    return all_passed


def test_mermaid_output():
    """
    测试Mermaid流程图生成（含条件分支）
    """
    print("\n测试3: Mermaid流程图生成")
    try:
        workflow = create_wealth_advisor_workflow()
        mermaid = workflow.get_graph().draw_mermaid()
        assert mermaid is not None, "Mermaid图生成失败"
        assert len(mermaid) > 0, "Mermaid图为空"
        # 检查关键节点
        for node in ["assess", "reactive", "collect_data", "analyze", "recommend", "respond"]:
            assert node in mermaid, f"节点 {node} 未在流程图中"
        print("  [PASS] Mermaid流程图生成成功，包含所有关键节点")
        return True
    except Exception as e:
        print(f"  [FAIL] Mermaid流程图生成失败: {str(e)}")
        return False


def test_api_key():
    """
    测试API密钥是否已配置
    """
    print("\n测试4: API密钥检查")
    api_key = os.getenv('DASHSCOPE_API_KEY', '')
    if api_key:
        print("  [PASS] DASHSCOPE_API_KEY 已配置")
        return True
    else:
        print("  [WARN] DASHSCOPE_API_KEY 未配置，跳过LLM调用测试")
        return False


def test_reactive_query():
    """
    测试反应式查询（简单快速查询）
    """
    print("\n测试5: 反应式查询 - 上证指数查询")
    api_key = os.getenv('DASHSCOPE_API_KEY', '')
    if not api_key:
        print("  [SKIP] API密钥未配置，跳过此测试")
        return None

    try:
        from hybrid_wealth_advisor_langgraph import run_wealth_advisor
        result = run_wealth_advisor("今天上证指数的表现如何？", "customer1")

        if result.get("error"):
            print(f"  [FAIL] 反应式查询出错: {result['error']}")
            return False

        final_response = result.get("final_response", "")
        if final_response and len(final_response) > 0:
            print(f"  [PASS] 反应式查询成功，响应长度: {len(final_response)}字符")
            return True
        else:
            print("  [FAIL] 反应式查询未生成响应")
            return False
    except Exception as e:
        print(f"  [FAIL] 反应式查询异常: {str(e)}")
        return False


def test_deliberative_query():
    """
    测试深思熟虑查询（深度分析查询）
    """
    print("\n测试6: 深思熟虑查询 - 投资组合优化")
    api_key = os.getenv('DASHSCOPE_API_KEY', '')
    if not api_key:
        print("  [SKIP] API密钥未配置，跳过此测试")
        return None

    try:
        from hybrid_wealth_advisor_langgraph import run_wealth_advisor
        result = run_wealth_advisor(
            "我应该如何调整投资组合以应对可能的经济衰退？",
            "customer1"
        )

        if result.get("error"):
            print(f"  [FAIL] 深思熟虑查询出错: {result['error']}")
            return False

        final_response = result.get("final_response", "")
        processing_mode = result.get("processing_mode", "未知")
        if final_response and len(final_response) > 0:
            print(f"  [PASS] 深思熟虑查询成功 (模式: {processing_mode})，响应长度: {len(final_response)}字符")
            return True
        else:
            print("  [FAIL] 深思熟虑查询未生成响应")
            return False
    except Exception as e:
        print(f"  [FAIL] 深思熟虑查询异常: {str(e)}")
        return False


def test_shanghai_index_tool():
    """
    测试上证指数查询工具（模拟数据）
    """
    print("\n测试7: 上证指数查询工具")
    try:
        from hybrid_wealth_advisor_langgraph import query_shanghai_index
        result = query_shanghai_index("")
        assert "上证指数" in result, "工具返回结果缺少'上证指数'"
        # 真实数据成功时包含"涨跌幅"，降级模拟数据时包含"模拟数据"
        has_real_data = "涨跌幅" in result
        has_fallback = "模拟数据" in result
        assert has_real_data or has_fallback, "工具返回结果格式异常"
        print(f"  [PASS] 工具返回: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] 工具调用失败: {str(e)}")
        return False


def main():
    """
    主测试函数
    """
    print("=" * 60)
    print("投顾AI助手（混合式）- 测试套件")
    print("=" * 60)

    results = []
    results.append(("工作流图创建", test_workflow_creation()))
    results.append(("客户画像数据", test_customer_profiles()))
    results.append(("Mermaid流程图", test_mermaid_output()))
    api_ok = test_api_key()
    results.append(("API密钥检查", api_ok))
    results.append(("上证指数查询工具", test_shanghai_index_tool()))
    results.append(("反应式查询", test_reactive_query()))
    results.append(("深思熟虑查询", test_deliberative_query()))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    for name, result in results:
        status = "PASS" if result is True else ("FAIL" if result is False else "SKIP")
        print(f"  [{status}] {name}")

    print(f"\n总计: {passed} 通过, {failed} 失败, {skipped} 跳过")
    print("=" * 60)


if __name__ == "__main__":
    main()
