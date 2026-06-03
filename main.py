#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能投顾AI助手系统 - 统一入口

整合三个子系统的统一启动脚本：
1. 私募基金运作指引问答助手（反应式）
2. 智能投研助手（深思熟虑）
3. 投顾AI助手（混合式）
"""

import os
import sys

# 将项目根目录和各子目录添加到模块搜索路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "01-私募基金运作指引问答助手（反应式）"))
sys.path.insert(0, os.path.join(BASE_DIR, "02-智能投研助手（深思熟虑）"))
sys.path.insert(0, os.path.join(BASE_DIR, "03-投顾AI助手（混合式）"))


def check_api_key():
    """
    检查API密钥是否已配置
    :return: True表示已配置，False表示未配置
    """
    api_key = os.getenv('DASHSCOPE_API_KEY', '')
    if not api_key:
        print("=" * 60)
        print("错误：未检测到 DASHSCOPE_API_KEY 环境变量")
        print("请先设置API密钥：")
        print("  Windows: set DASHSCOPE_API_KEY=your-api-key")
        print("  Linux/Mac: export DASHSCOPE_API_KEY=your-api-key")
        print("=" * 60)
        return False
    return True


def run_fund_qa():
    """
    启动私募基金问答助手（反应式）
    """
    if not check_api_key():
        return
    from fund_qa_langgraph_v2 import FundQAAssistant

    assistant = FundQAAssistant()
    print("\n" + "=" * 60)
    print("  私募基金运作指引问答助手（反应式）")
    print("  输入 exit 退出，输入 help 查看帮助")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("请输入您的私募基金问题：")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if query.strip().lower() == "exit":
            print("已退出。")
            break
        if query.strip().lower() == "help":
            print("支持以下类型的提问：")
            print("  - 标准问法：私募基金的合格投资者标准是什么？")
            print("  - 关键词查询：合格投资者")
            print("  - 口语化表达：想投私募基金需要什么条件？")
            print("  - 场景化提问：公司投资私募基金有什么要求？")
            continue
        if not query.strip():
            continue
        answer = assistant.process_query(query)
        print(f"\n{answer}\n")


def run_research_assistant():
    """
    启动智能投研助手（深思熟虑）
    """
    if not check_api_key():
        return
    from deliberative_research_langgraph import run_research_agent

    print("\n" + "=" * 60)
    print("  智能投研助手（深思熟虑）")
    print("=" * 60 + "\n")

    topic = input("请输入研究主题 (例如: 新能源汽车行业投资机会): ").strip()
    if not topic:
        print("研究主题不能为空。")
        return
    industry = input("请输入行业焦点 (例如: 电动汽车制造、电池技术): ").strip()
    if not industry:
        industry = "综合"
    horizon = input("请输入时间范围 [短期/中期/长期]: ").strip()
    if horizon not in ("短期", "中期", "长期"):
        horizon = "中期"

    print(f"\n智能投研助手开始工作...\n")
    try:
        result = run_research_agent(topic, industry, horizon)
        if result.get("error"):
            print(f"\n发生错误: {result['error']}")
        else:
            print("\n=== 最终研究报告 ===\n")
            print(result.get("final_report", "未生成报告"))
    except Exception as e:
        print(f"\n运行过程中发生错误: {str(e)}")


def run_wealth_advisor():
    """
    启动投顾AI助手（混合式）
    """
    if not check_api_key():
        return
    from hybrid_wealth_advisor_langgraph import run_wealth_advisor, SAMPLE_CUSTOMER_PROFILES

    print("\n" + "=" * 60)
    print("  投顾AI助手（混合式）")
    print("=" * 60 + "\n")

    # 示例查询
    sample_queries = [
        "今天上证指数的表现如何？",
        "请解释一下什么是ETF？",
        "根据当前市场情况，我应该如何调整投资组合以应对可能的经济衰退？",
        "考虑到我的退休目标，请评估我当前的投资策略并提供优化建议。",
        "我想为子女准备教育金，请帮我设计一个10年期的投资计划。",
    ]

    print("请选择一个示例查询或输入您自己的查询:\n")
    for i, query in enumerate(sample_queries, 1):
        print(f"  {i}. {query}")
    print("  0. 输入自定义查询")

    choice = input("\n请输入选项数字(0-5): ").strip()
    if choice == "0":
        user_query = input("请输入您的查询: ").strip()
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sample_queries):
                user_query = sample_queries[idx]
            else:
                print("无效选择，使用默认查询")
                user_query = sample_queries[0]
        except ValueError:
            print("无效输入，使用默认查询")
            user_query = sample_queries[0]

    if not user_query:
        print("查询不能为空。")
        return

    # 选择客户画像
    print("\n选择客户画像:")
    for cid, profile in SAMPLE_CUSTOMER_PROFILES.items():
        print(f"  {cid}: {profile['risk_tolerance']}投资者 (组合价值: {profile['portfolio_value']:,.0f}元)")
    customer_choice = input("请输入客户ID (默认customer1): ").strip()
    customer_id = customer_choice if customer_choice in SAMPLE_CUSTOMER_PROFILES else "customer1"

    print(f"\n正在处理...\n")
    try:
        from datetime import datetime
        start_time = datetime.now()
        result = run_wealth_advisor(user_query, customer_id)
        end_time = datetime.now()

        if result.get("error"):
            print(f"处理过程中发生错误: {result['error']}")
            print(f"\n最终响应: {result.get('final_response', '未能生成响应')}")
        else:
            process_mode = result.get("processing_mode", "未知")
            if process_mode == "reactive":
                print("【处理模式: 反应式】- 快速响应简单查询")
            else:
                print("【处理模式: 深思熟虑】- 深度分析复杂查询")

            print("\n=== 响应结果 ===\n")
            print(result.get("final_response", "未生成响应"))

        process_time = (end_time - start_time).total_seconds()
        print(f"\n处理用时: {process_time:.2f}秒")
    except Exception as e:
        print(f"\n运行过程中发生意外错误: {str(e)}")


def main():
    """
    主入口函数，提供子系统选择菜单
    """
    print("\n" + "=" * 60)
    print("       智能投顾AI助手系统")
    print("=" * 60)
    print("\n请选择要启动的子系统：\n")
    print("  1. 私募基金运作指引问答助手（反应式）")
    print("     - 22条私募基金规则智能问答")
    print("     - 适合：合规查询、规则咨询")
    print()
    print("  2. 智能投研助手（深思熟虑）")
    print("     - 五阶段深度投研分析")
    print("     - 适合：行业研究、投资报告生成")
    print()
    print("  3. 投顾AI助手（混合式）")
    print("     - 智能路由+个性化投资建议")
    print("     - 适合：财富管理、投资规划")
    print()
    print("  0. 退出")
    print()

    while True:
        choice = input("请输入选项(0-3): ").strip()
        if choice == "1":
            run_fund_qa()
            break
        elif choice == "2":
            run_research_assistant()
            break
        elif choice == "3":
            run_wealth_advisor()
            break
        elif choice == "0":
            print("已退出。")
            break
        else:
            print("无效选项，请重新输入(0-3)。")


if __name__ == "__main__":
    main()
