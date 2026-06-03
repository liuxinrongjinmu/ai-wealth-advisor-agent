#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能投研助手（深思熟虑）- 测试脚本

验证五阶段投研流程的完整性和正确性
"""

import os
import sys

# 添加模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deliberative_research_langgraph import (
    create_research_agent_workflow,
    ResearchAgentState
)


def test_workflow_creation():
    """
    测试工作流图创建是否成功
    """
    print("测试1: 工作流图创建")
    try:
        workflow = create_research_agent_workflow()
        assert workflow is not None, "工作流创建失败"
        print("  [PASS] 工作流图创建成功")
        return True
    except Exception as e:
        print(f"  [FAIL] 工作流图创建失败: {str(e)}")
        return False


def test_initial_state():
    """
    测试初始状态结构是否完整
    """
    print("\n测试2: 初始状态结构")
    required_fields = [
        "research_topic", "industry_focus", "time_horizon",
        "perception_data", "world_model", "reasoning_plans",
        "selected_plan", "final_report", "current_phase", "error"
    ]
    try:
        initial_state = {
            "research_topic": "测试主题",
            "industry_focus": "测试行业",
            "time_horizon": "中期",
            "perception_data": None,
            "world_model": None,
            "reasoning_plans": None,
            "selected_plan": None,
            "final_report": None,
            "current_phase": "perception",
            "error": None
        }
        for field in required_fields:
            assert field in initial_state, f"缺少字段: {field}"
        print("  [PASS] 初始状态结构完整")
        return True
    except AssertionError as e:
        print(f"  [FAIL] {str(e)}")
        return False


def test_mermaid_output():
    """
    测试Mermaid流程图生成
    """
    print("\n测试3: Mermaid流程图生成")
    try:
        workflow = create_research_agent_workflow()
        mermaid = workflow.get_graph().draw_mermaid()
        assert mermaid is not None, "Mermaid图生成失败"
        assert len(mermaid) > 0, "Mermaid图为空"
        # 检查关键节点是否在图中
        for node in ["perception", "modeling", "reasoning", "decision", "report"]:
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


def test_full_workflow():
    """
    测试完整的五阶段工作流（需要API密钥）
    """
    print("\n测试5: 完整五阶段工作流（需API密钥）")
    api_key = os.getenv('DASHSCOPE_API_KEY', '')
    if not api_key:
        print("  [SKIP] API密钥未配置，跳过此测试")
        return None

    try:
        workflow = create_research_agent_workflow()
        initial_state = {
            "research_topic": "人工智能行业投资机会",
            "industry_focus": "大语言模型、计算机视觉",
            "time_horizon": "中期",
            "perception_data": None,
            "world_model": None,
            "reasoning_plans": None,
            "selected_plan": None,
            "final_report": None,
            "current_phase": "perception",
            "error": None
        }
        result = workflow.invoke(initial_state)

        # 检查是否有错误
        if result.get("error"):
            print(f"  [FAIL] 工作流执行出错: {result['error']}")
            return False

        # 检查各阶段输出
        checks = [
            ("perception_data", result.get("perception_data") is not None),
            ("world_model", result.get("world_model") is not None),
            ("reasoning_plans", result.get("reasoning_plans") is not None),
            ("selected_plan", result.get("selected_plan") is not None),
            ("final_report", result.get("final_report") is not None and len(result["final_report"]) > 0),
        ]

        all_passed = True
        for name, passed in checks:
            if passed:
                print(f"  [PASS] {name} 已生成")
            else:
                print(f"  [FAIL] {name} 未生成")
                all_passed = False

        return all_passed
    except Exception as e:
        print(f"  [FAIL] 工作流执行异常: {str(e)}")
        return False


def main():
    """
    主测试函数
    """
    print("=" * 60)
    print("智能投研助手（深思熟虑）- 测试套件")
    print("=" * 60)

    results = []
    results.append(("工作流图创建", test_workflow_creation()))
    results.append(("初始状态结构", test_initial_state()))
    results.append(("Mermaid流程图", test_mermaid_output()))
    api_ok = test_api_key()
    results.append(("API密钥检查", api_ok))
    results.append(("完整五阶段工作流", test_full_workflow()))

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
