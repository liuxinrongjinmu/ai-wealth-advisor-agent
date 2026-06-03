#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能投顾AI助手系统 - Web API服务

基于FastAPI提供的RESTful API服务，整合三个子系统的功能
启动方式：python web_api.py
访问地址：http://localhost:8000
API文档：http://localhost:8000/docs
"""

import os
import sys
import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 将项目根目录和各子目录添加到模块搜索路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "01-私募基金运作指引问答助手（反应式）"))
sys.path.insert(0, os.path.join(BASE_DIR, "02-智能投研助手（深思熟虑）"))
sys.path.insert(0, os.path.join(BASE_DIR, "03-投顾AI助手（混合式）"))

app = FastAPI(
    title="智能投顾AI助手系统",
    description="基于LangChain+LangGraph的智能投顾AI助手系统API，包含私募基金问答、投研分析、财富顾问三大功能",
    version="1.0.0"
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 请求/响应模型 ==========

class FundQARequest(BaseModel):
    """私募基金问答请求"""
    query: str = Field(..., description="用户查询问题", min_length=1, max_length=500)


class FundQAResponse(BaseModel):
    """私募基金问答响应"""
    query: str = Field(..., description="用户查询问题")
    answer: str = Field(..., description="回答内容")
    category: str = Field(default="", description="规则分类")


class ResearchRequest(BaseModel):
    """投研分析请求"""
    topic: str = Field(..., description="研究主题", min_length=1, max_length=200)
    industry: str = Field(default="综合", description="行业焦点")
    horizon: str = Field(default="中期", description="时间范围：短期/中期/长期")


class ResearchResponse(BaseModel):
    """投研分析响应"""
    topic: str = Field(..., description="研究主题")
    industry: str = Field(..., description="行业焦点")
    horizon: str = Field(..., description="时间范围")
    report: str = Field(default="", description="研究报告内容")
    error: Optional[str] = Field(default=None, description="错误信息")


class WealthAdvisorRequest(BaseModel):
    """财富顾问请求"""
    query: str = Field(..., description="用户查询问题", min_length=1, max_length=500)
    customer_id: str = Field(default="customer1", description="客户ID：customer1(平衡型)/customer2(进取型)")


class WealthAdvisorResponse(BaseModel):
    """财富顾问响应"""
    query: str = Field(..., description="用户查询问题")
    response: str = Field(..., description="响应内容")
    processing_mode: str = Field(default="", description="处理模式：reactive/deliberative")
    error: Optional[str] = Field(default=None, description="错误信息")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    api_key_configured: bool = Field(..., description="API密钥是否已配置")


# ========== 全局实例（懒加载） ==========

_fund_qa_assistant = None
_research_workflow = None


def get_fund_qa_assistant():
    """
    获取私募基金问答助手实例（懒加载）
    :return: FundQAAssistant实例
    """
    global _fund_qa_assistant
    if _fund_qa_assistant is None:
        from fund_qa_langgraph_v2 import FundQAAssistant
        _fund_qa_assistant = FundQAAssistant()
    return _fund_qa_assistant


def get_research_workflow():
    """
    获取投研助手工作流实例（懒加载）
    :return: 编译后的StateGraph实例
    """
    global _research_workflow
    if _research_workflow is None:
        from deliberative_research_langgraph import create_research_agent_workflow
        _research_workflow = create_research_agent_workflow()
    return _research_workflow


# ========== API路由 ==========

@app.get("/", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """
    健康检查接口，返回服务状态和API密钥配置情况
    """
    api_key = os.getenv('DASHSCOPE_API_KEY', '')
    return HealthResponse(
        status="running",
        api_key_configured=bool(api_key)
    )


@app.post("/api/fund-qa", response_model=FundQAResponse, tags=["私募基金问答"])
async def fund_qa(request: FundQARequest):
    """
    私募基金运作指引问答接口
    支持22条私募基金核心规则的智能匹配问答
    """
    try:
        assistant = get_fund_qa_assistant()
        # 异步执行同步的问答处理，避免阻塞事件循环
        answer = await asyncio.to_thread(assistant.process_query, request.query)
        # 提取分类前缀
        category = ""
        if answer.startswith("【") and "】" in answer:
            end_idx = answer.index("】")
            category = answer[1:end_idx]
        return FundQAResponse(
            query=request.query,
            answer=answer,
            category=category
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理查询时出错: {str(e)}")


@app.post("/api/research", response_model=ResearchResponse, tags=["投研分析"])
async def research(request: ResearchRequest):
    """
    智能投研分析接口
    五阶段深度分析流程：感知→建模→推理→决策→报告
    """
    try:
        workflow = get_research_workflow()
        initial_state = {
            "research_topic": request.topic,
            "industry_focus": request.industry,
            "time_horizon": request.horizon,
            "perception_data": None,
            "world_model": None,
            "reasoning_plans": None,
            "selected_plan": None,
            "final_report": None,
            "current_phase": "perception",
            "error": None
        }
        # 异步执行同步的工作流，避免阻塞事件循环
        result = await asyncio.to_thread(workflow.invoke, initial_state)

        if result.get("error"):
            return ResearchResponse(
                topic=request.topic,
                industry=request.industry,
                horizon=request.horizon,
                error=result["error"]
            )

        return ResearchResponse(
            topic=request.topic,
            industry=request.industry,
            horizon=request.horizon,
            report=result.get("final_report", "未生成报告")
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"投研分析出错: {str(e)}")


@app.post("/api/wealth-advisor", response_model=WealthAdvisorResponse, tags=["财富顾问"])
async def wealth_advisor(request: WealthAdvisorRequest):
    """
    混合式财富顾问接口
    智能路由：简单查询快速响应，复杂查询深度分析
    """
    try:
        from hybrid_wealth_advisor_langgraph import run_wealth_advisor
        # 异步执行同步的顾问处理，避免阻塞事件循环
        result = await asyncio.to_thread(run_wealth_advisor, request.query, request.customer_id)

        if result.get("error"):
            return WealthAdvisorResponse(
                query=request.query,
                response=result.get("final_response", "处理失败"),
                processing_mode=result.get("processing_mode", "未知"),
                error=result["error"]
            )

        return WealthAdvisorResponse(
            query=request.query,
            response=result.get("final_response", "未生成响应"),
            processing_mode=result.get("processing_mode", "未知")
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"财富顾问处理出错: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  智能投顾AI助手系统 - Web API服务")
    print("  API文档: http://localhost:8002/docs")
    print("  健康检查: http://localhost:8002/")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8002)
