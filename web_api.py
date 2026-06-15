#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
智能投顾AI助手系统 - Web API服务

基于FastAPI提供的RESTful API服务，整合三个子系统的功能
启动方式：python web_api.py
访问地址：http://localhost:8002
API文档：http://localhost:8002/docs
"""

import os
import sys
import json
import asyncio
import logging
import time
import threading
from typing import Optional, List, Tuple
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from logging.handlers import RotatingFileHandler

# 将项目根目录和各子目录添加到模块搜索路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "01-私募基金运作指引问答助手（反应式）"))
sys.path.insert(0, os.path.join(BASE_DIR, "02-智能投研助手（深思熟虑）"))
sys.path.insert(0, os.path.join(BASE_DIR, "03-投顾AI助手（混合式）"))

# 导入共享数据库模块（替换原有JSON DataStore）
import db


# ========== 统一配置管理 ==========

class Settings(BaseSettings):
    """
    应用配置（支持环境变量和.env文件）
    """
    # 服务配置
    app_host: str = "0.0.0.0"
    app_port: int = 8002
    app_name: str = "智能投顾AI助手系统"
    app_version: str = "1.0.0"
    debug: bool = False

    # API认证（逗号分隔的合法API Key列表，为空则不启用认证）
    api_keys: str = ""
    api_key_name: str = "X-API-Key"

    # CORS配置（逗号分隔的允许域名列表）
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"

    # 速率限制（每分钟最大请求数）
    rate_limit_per_minute: int = 30

    # 请求超时（秒）- 投研分析5阶段LLM调用需要较长时间
    request_timeout: int = 300

    # 日志配置
    log_file: str = "logs/app.log"
    log_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5

    # LLM配置
    dashscope_api_key: str = ""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


settings = Settings()


# ========== 数据存储（SQLite数据库，已替换原有JSON DataStore） ==========
# 所有数据持久化通过 db.py 模块实现，在模块导入时自动初始化数据库表
# 包括：customer_profiles、conversations、research_reports 三张表


# ========== 日志配置（持久化） ==========

def setup_logging():
    """
    配置日志系统，同时输出到控制台和文件
    """
    log_dir = os.path.dirname(settings.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level = logging.DEBUG if settings.debug else logging.INFO

    # 根日志配置
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 控制台Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    # 文件Handler（RotatingFileHandler，自动轮转）
    file_handler = RotatingFileHandler(
        settings.log_file,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)

    # 降低第三方库日志级别
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("日志系统初始化完成，日志文件: %s", settings.log_file)


setup_logging()
logger = logging.getLogger(__name__)


# ========== API认证 ==========

api_key_header = APIKeyHeader(name=settings.api_key_name, auto_error=False)


def _get_valid_api_keys() -> List[str]:
    """
    获取合法的API Key列表
    :return: API Key列表
    """
    keys_str = settings.api_keys.strip()
    if not keys_str:
        return []
    return [k.strip() for k in keys_str.split(",") if k.strip()]


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    验证API Key（如果配置了api_keys则强制校验，否则跳过）
    :param api_key: 请求头中的API Key
    :return: 验证通过的API Key
    :raises HTTPException: 认证失败时抛出401
    """
    valid_keys = _get_valid_api_keys()
    # 未配置API Key则不启用认证
    if not valid_keys:
        return "anonymous"
    if not api_key or api_key not in valid_keys:
        logger.warning("API认证失败: %s", "未提供Key" if not api_key else "Key无效")
        raise HTTPException(
            status_code=401,
            detail="无效的API Key，请在请求头中提供有效的X-API-Key"
        )
    return api_key


# ========== 简易速率限制（基于IP的内存计数器） ==========

class RateLimiter:
    """
    基于IP的简易速率限制器（内存级，单进程适用）
    生产环境建议替换为Redis实现
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        """
        初始化速率限制器
        :param max_requests: 窗口期内最大请求数
        :param window_seconds: 窗口期（秒）
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict = {}  # {ip: [timestamp1, timestamp2, ...]}

    def is_allowed(self, client_ip: str) -> bool:
        """
        检查请求是否被允许
        :param client_ip: 客户端IP
        :return: True表示允许，False表示超限
        """
        now = time.time()
        if client_ip not in self._requests:
            self._requests[client_ip] = [now]
            return True

        # 清理过期记录
        self._requests[client_ip] = [
            t for t in self._requests[client_ip]
            if now - t < self.window_seconds
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            return False

        self._requests[client_ip].append(now)
        return True


rate_limiter = RateLimiter(
    max_requests=settings.rate_limit_per_minute,
    window_seconds=60
)


# ========== FastAPI应用 ==========

app = FastAPI(
    title=settings.app_name,
    description="基于LangChain+LangGraph的智能投顾AI助手系统API，包含私募基金问答、投研分析、财富顾问三大功能",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS配置（通过环境变量控制）
cors_origins_list = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    max_age=3600,
)


# ========== 请求/响应模型 ==========

class FundQARequest(BaseModel):
    """私募基金问答请求"""
    query: str = Field(..., description="用户查询问题", min_length=1, max_length=500)
    thread_id: Optional[str] = Field(None, description="会话线程ID，用于多轮对话上下文记忆")


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
    thread_id: Optional[str] = Field(None, description="会话线程ID，用于多轮对话上下文记忆")


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
    thread_id: Optional[str] = Field(default=None, description="对话线程ID，传入此参数启用多轮对话记忆")


class WealthAdvisorResponse(BaseModel):
    """财富顾问响应"""
    query: str = Field(..., description="用户查询问题")
    response: str = Field(..., description="响应内容")
    processing_mode: str = Field(default="", description="处理模式：reactive/deliberative")
    error: Optional[str] = Field(default=None, description="错误信息")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    api_key_configured: bool = Field(..., description="LLM API密钥是否已配置")
    auth_enabled: bool = Field(..., description="API认证是否启用")
    version: str = Field(..., description="系统版本")


class CustomerProfileCreate(BaseModel):
    """创建客户画像请求"""
    customer_id: str = Field(..., description="客户ID", min_length=1, max_length=50)
    name: str = Field(..., description="客户姓名", min_length=1, max_length=50)
    risk_tolerance: str = Field(default="平衡型", description="风险偏好：保守型/稳健型/平衡型/成长型/进取型")
    investment_experience: str = Field(default="一般", description="投资经验：无/一般/丰富")
    investment_horizon: str = Field(default="中期", description="投资期限：短期/中期/长期")
    investment_amount: str = Field(default="50-100万", description="投资金额范围")
    focus_sectors: str = Field(default="科技,消费", description="关注行业（逗号分隔）")
    age: int = Field(default=35, description="年龄")
    occupation: str = Field(default="", description="职业")
    annual_income: str = Field(default="", description="年收入")


class CustomerProfileResponse(BaseModel):
    """客户画像响应"""
    customer_id: str
    name: str
    risk_tolerance: str
    investment_experience: str
    investment_horizon: str
    investment_amount: str
    focus_sectors: str
    created_at: Optional[str] = None


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


# ========== API路由（v1版本） ==========

# 导入认证模块
import auth as auth_module
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

jwt_scheme = HTTPBearer(auto_error=False)


async def get_jwt_user(
    credentials: HTTPAuthorizationCredentials = Security(jwt_scheme),
    api_key: str = Depends(verify_api_key),
) -> Optional[Tuple[str, int]]:
    """
    从JWT或API Key中获取当前用户
    优先使用JWT认证，失败时回退到API Key（保持向后兼容）
    :param credentials: JWT Bearer token
    :param api_key: API Key（兼容旧版）
    :return: (username, user_id) 或 None
    """
    if credentials:
        user = auth_module.get_current_user(credentials.credentials)
        if user:
            return user
    return None  # API Key认证不返回用户身份


# ========== 认证端点 ==========

class RegisterRequest(BaseModel):
    """用户注册请求"""
    username: str = Field(..., description="用户名", min_length=3, max_length=30)
    password: str = Field(..., description="密码", min_length=6, max_length=50)
    display_name: str = Field(default="", description="显示名称", max_length=50)
    email: str = Field(default="", description="邮箱", max_length=100)


class LoginRequest(BaseModel):
    """用户登录请求"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class TokenResponse(BaseModel):
    """Token响应"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: dict


class RefreshRequest(BaseModel):
    """刷新Token请求"""
    refresh_token: str = Field(..., description="刷新令牌")


@app.post("/api/v1/auth/register", tags=["认证"])
async def register(request: RegisterRequest):
    """
    用户注册接口
    创建新用户并返回JWT令牌
    """
    existing = db.get_user_by_username(request.username)
    if existing:
        raise HTTPException(status_code=409, detail="用户名已存在")

    password_hash = auth_module.hash_password(request.password)
    user = db.create_user(
        username=request.username,
        password_hash=password_hash,
        display_name=request.display_name or request.username,
        email=request.email,
    )
    if user is None:
        raise HTTPException(status_code=500, detail="用户创建失败")

    access_token = auth_module.create_access_token(request.username, user["id"])
    refresh_token = auth_module.create_refresh_token(request.username, user["id"])
    logger.info("新用户注册: %s", request.username)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user={"id": user["id"], "username": user["username"], "display_name": user["display_name"]},
    )


@app.post("/api/v1/auth/login", tags=["认证"])
async def login(request: LoginRequest):
    """
    用户登录接口
    验证用户名密码，返回JWT令牌
    """
    user = db.get_user_by_username(request.username)
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    if not auth_module.verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    access_token = auth_module.create_access_token(request.username, user["id"])
    refresh_token = auth_module.create_refresh_token(request.username, user["id"])
    logger.info("用户登录: %s", request.username)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user={"id": user["id"], "username": user["username"], "display_name": user["display_name"]},
    )


@app.post("/api/v1/auth/refresh", tags=["认证"])
async def refresh_token(request: RefreshRequest):
    """
    刷新访问令牌
    使用refresh_token获取新的access_token
    """
    payload = auth_module.decode_token(request.refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="无效的刷新令牌")

    username = payload.get("sub")
    user_id = payload.get("user_id")
    if not username or not user_id:
        raise HTTPException(status_code=401, detail="无效的令牌内容")

    new_access = auth_module.create_access_token(username, user_id)
    new_refresh = auth_module.create_refresh_token(username, user_id)

    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        user={"id": user_id, "username": username},
    )


@app.get("/api/v1/auth/me", tags=["认证"])
async def get_me(user: Tuple[str, int] = Depends(get_jwt_user)):
    """
    获取当前登录用户信息
    需要JWT认证
    """
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录")
    username, user_id = user
    user_info = db.get_user_by_id(user_id)
    if user_info is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user_info


# ========== 业务API端点 ==========

@app.get("/", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """
    健康检查接口，返回服务状态和配置情况
    """
    api_key = os.getenv('DASHSCOPE_API_KEY', '') or settings.dashscope_api_key
    valid_keys = _get_valid_api_keys()
    return HealthResponse(
        status="running",
        api_key_configured=bool(api_key),
        auth_enabled=bool(valid_keys),
        version=settings.app_version
    )


@app.post("/api/v1/fund-qa", response_model=FundQAResponse, tags=["私募基金问答"])
async def fund_qa(request: FundQARequest, api_key: str = Depends(verify_api_key)):
    """
    私募基金运作指引问答接口
    支持22条私募基金核心规则的智能匹配问答
    支持多轮对话：传入 thread_id 可保持上下文记忆
    """
    try:
        # 多轮对话上下文：获取同一线程的历史对话
        context_messages = []
        if request.thread_id:
            try:
                history = db.get_conversation_history(request.thread_id)
                # 取最近4条消息作为上下文
                context_messages = [(m["role"], m["content"]) for m in history[-4:]]
            except Exception as e:
                logger.debug("获取对话上下文失败: %s", str(e))

        assistant = get_fund_qa_assistant()
        # 构建带上下文的查询
        if context_messages:
            context_str = "\n".join([f"[{r}]: {c[:200]}" for r, c in context_messages])
            full_query = f"对话历史：\n{context_str}\n\n当前问题：{request.query}"
        else:
            full_query = request.query

        answer = await asyncio.wait_for(
            asyncio.to_thread(assistant.process_query, full_query),
            timeout=settings.request_timeout
        )
        # 提取分类前缀
        category = ""
        if answer.startswith("【") and "】" in answer:
            end_idx = answer.index("】")
            category = answer[1:end_idx]
        # 持久化对话记录（复用传入的 thread_id 或生成新ID）
        thread_id = request.thread_id or f"fund-qa-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        db.save_conversation(
            thread_id=thread_id,
            role="user", content=request.query, tab_type="fund-qa"
        )
        db.save_conversation(
            thread_id=thread_id,
            role="assistant", content=answer, tab_type="fund-qa"
        )
        return FundQAResponse(
            query=request.query,
            answer=answer,
            category=category
        )
    except asyncio.TimeoutError:
        logger.error("私募基金问答超时: query=%s", request.query[:50])
        raise HTTPException(status_code=504, detail="私募基金问答处理超时，请稍后重试")
    except ValueError as e:
        logger.error("私募基金问答ValueError: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("私募基金问答异常: %s", str(e))
        raise HTTPException(status_code=500, detail=f"处理查询时出错: {str(e)}")


@app.post("/api/v1/research", response_model=ResearchResponse, tags=["投研分析"])
async def research(request: ResearchRequest, api_key: str = Depends(verify_api_key)):
    """
    智能投研分析接口
    五阶段深度分析流程：感知→建模→推理→决策→报告
    支持多轮对话：传入 thread_id 可保持上下文记忆
    """
    try:
        # 多轮对话上下文：获取同一线程的历史对话
        context_messages = []
        if request.thread_id:
            try:
                history = db.get_conversation_history(request.thread_id)
                context_messages = [(m["role"], m["content"]) for m in history[-4:]]
            except Exception as e:
                logger.debug("获取投研上下文失败: %s", str(e))

        # 构建带上下文的查询
        if context_messages:
            context_str = "\n".join([f"[{r}]: {c[:200]}" for r, c in context_messages])
            full_topic = f"此前对话：\n{context_str}\n\n当前研究主题：{request.topic}"
        else:
            full_topic = request.topic

        workflow = get_research_workflow()
        initial_state = {
            "research_topic": full_topic,
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
        result = await asyncio.wait_for(
            asyncio.to_thread(workflow.invoke, initial_state),
            timeout=settings.request_timeout
        )

        if result.get("error"):
            return ResearchResponse(
                topic=request.topic,
                industry=request.industry,
                horizon=request.horizon,
                error=result["error"]
            )

        report_content = result.get("final_report", "未生成报告")
        # 持久化投研报告到数据库
        db.save_report(
            topic=request.topic,
            report=report_content,
            industry=request.industry,
            horizon=request.horizon,
            status="completed"
        )
        # 持久化对话记录（复用传入的 thread_id 或生成新ID）
        thread_id = request.thread_id or f"research-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        db.save_conversation(thread_id=thread_id, role="user", content=request.topic, tab_type="research")
        db.save_conversation(thread_id=thread_id, role="assistant", content=report_content[:500], tab_type="research")

        return ResearchResponse(
            topic=request.topic,
            industry=request.industry,
            horizon=request.horizon,
            report=report_content
        )
    except asyncio.TimeoutError:
        logger.error("投研分析超时: topic=%s", request.topic)
        raise HTTPException(status_code=504, detail="投研分析处理超时，请稍后重试")
    except ValueError as e:
        logger.error("投研分析ValueError: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("投研分析异常: %s", str(e))
        raise HTTPException(status_code=500, detail=f"投研分析出错: {str(e)}")


@app.post("/api/v1/wealth-advisor", response_model=WealthAdvisorResponse, tags=["财富顾问"])
async def wealth_advisor(request: WealthAdvisorRequest, api_key: str = Depends(verify_api_key)):
    """
    混合式财富顾问接口
    智能路由：简单查询快速响应，复杂查询深度分析
    """
    try:
        from hybrid_wealth_advisor_langgraph import run_wealth_advisor
        result = await asyncio.wait_for(
            asyncio.to_thread(run_wealth_advisor, request.query, request.customer_id, request.thread_id),
            timeout=settings.request_timeout
        )

        if result.get("error"):
            return WealthAdvisorResponse(
                query=request.query,
                response=result.get("final_response", "处理失败"),
                processing_mode=result.get("processing_mode", "未知"),
                error=result["error"]
            )

        response_text = result.get("final_response", "未生成响应")
        mode = result.get("processing_mode", "未知")
        # 持久化对话记录
        thread_id = request.thread_id or f"wealth-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        db.save_conversation(thread_id=thread_id, role="user", content=request.query,
                           customer_id=request.customer_id, processing_mode=mode, tab_type="wealth-advisor")
        db.save_conversation(thread_id=thread_id, role="assistant", content=response_text,
                           customer_id=request.customer_id, processing_mode=mode, tab_type="wealth-advisor")

        return WealthAdvisorResponse(
            query=request.query,
            response=response_text,
            processing_mode=mode
        )
    except asyncio.TimeoutError:
        logger.error("财富顾问超时: query=%s", request.query[:50])
        raise HTTPException(status_code=504, detail="财富顾问处理超时，请稍后重试")
    except ValueError as e:
        logger.error("财富顾问ValueError: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("财富顾问异常: %s", str(e))
        raise HTTPException(status_code=500, detail=f"财富顾问处理出错: {str(e)}")


# ========== 客户管理 & 对话管理端点 ==========

@app.get("/api/v1/customers", tags=["客户管理"])
async def list_customers(api_key: str = Depends(verify_api_key)):
    """获取所有客户画像列表"""
    return db.get_all_profiles()


@app.get("/api/v1/customers/{customer_id}", tags=["客户管理"])
async def get_customer(customer_id: str, api_key: str = Depends(verify_api_key)):
    """获取指定客户画像"""
    profile = db.get_profile(customer_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"客户 {customer_id} 不存在")
    return profile


@app.post("/api/v1/customers", tags=["客户管理"])
async def create_customer(profile: CustomerProfileCreate, api_key: str = Depends(verify_api_key)):
    """创建客户画像"""
    existing = db.get_profile(profile.customer_id)
    if existing:
        raise HTTPException(status_code=409, detail=f"客户 {profile.customer_id} 已存在")
    new_profile = db.create_profile(profile.model_dump())
    logger.info("创建客户画像: %s (%s)", profile.customer_id, profile.name)
    return new_profile


@app.put("/api/v1/customers/{customer_id}", tags=["客户管理"])
async def update_customer(customer_id: str, profile: CustomerProfileCreate, api_key: str = Depends(verify_api_key)):
    """更新客户画像"""
    existing = db.get_profile(customer_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"客户 {customer_id} 不存在")
    updated = db.update_profile(customer_id, profile.model_dump())
    logger.info("更新客户画像: %s", customer_id)
    return updated


@app.delete("/api/v1/customers/{customer_id}", tags=["客户管理"])
async def delete_customer(customer_id: str, api_key: str = Depends(verify_api_key)):
    """删除客户画像"""
    if not db.delete_profile(customer_id):
        raise HTTPException(status_code=404, detail=f"客户 {customer_id} 不存在")
    logger.info("删除客户画像: %s", customer_id)
    return {"message": f"客户 {customer_id} 已删除"}


@app.get("/api/v1/conversations/{thread_id}", tags=["对话管理"])
async def get_conversation(thread_id: str, api_key: str = Depends(verify_api_key)):
    """获取对话历史"""
    history = db.get_conversation_history(thread_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"对话 {thread_id} 不存在")
    return history


@app.get("/api/v1/conversations", tags=["对话管理"])
async def list_conversation_threads(api_key: str = Depends(verify_api_key)):
    """获取所有对话线程列表"""
    return db.get_all_threads()


@app.get("/api/v1/conversations/{thread_id}", tags=["对话管理"])
async def get_thread_messages(thread_id: str, api_key: str = Depends(verify_api_key)):
    """获取指定线程的所有消息"""
    messages = db.get_conversation_history(thread_id)
    return messages


@app.delete("/api/v1/conversations/{thread_id}", tags=["对话管理"])
async def delete_conversation_thread(thread_id: str, api_key: str = Depends(verify_api_key)):
    """删除指定对话线程及其所有消息"""
    success = db.delete_thread(thread_id)
    if not success:
        raise HTTPException(status_code=404, detail="对话线程不存在")
    return {"message": "删除成功", "thread_id": thread_id}


@app.get("/api/v1/reports", tags=["投研报告"])
async def list_reports(api_key: str = Depends(verify_api_key)):
    """获取投研报告列表"""
    return db.get_reports()


@app.get("/api/v1/reports/{report_id}", tags=["投研报告"])
async def get_report_detail(report_id: int, api_key: str = Depends(verify_api_key)):
    """获取投研报告详情"""
    report = db.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"报告 {report_id} 不存在")
    return report


# ========== 兼容旧版API路径（重定向到v1） ==========

@app.post("/api/fund-qa", response_model=FundQAResponse, tags=["私募基金问答（兼容）"], deprecated=True)
async def fund_qa_legacy(request: FundQARequest, api_key: str = Depends(verify_api_key)):
    """兼容旧版API路径，建议使用 /api/v1/fund-qa"""
    return await fund_qa(request, api_key)


@app.post("/api/research", response_model=ResearchResponse, tags=["投研分析（兼容）"], deprecated=True)
async def research_legacy(request: ResearchRequest, api_key: str = Depends(verify_api_key)):
    """兼容旧版API路径，建议使用 /api/v1/research"""
    return await research(request, api_key)


@app.post("/api/wealth-advisor", response_model=WealthAdvisorResponse, tags=["财富顾问（兼容）"], deprecated=True)
async def wealth_advisor_legacy(request: WealthAdvisorRequest, api_key: str = Depends(verify_api_key)):
    """兼容旧版API路径，建议使用 /api/v1/wealth-advisor"""
    return await wealth_advisor(request, api_key)


# ========== SSE流式输出端点 ==========

@app.get("/api/v1/fund-qa/stream", tags=["私募基金问答（流式）"])
async def fund_qa_stream(query: str, api_key: str = Depends(verify_api_key)):
    """
    私募基金问答流式接口（SSE）
    逐步返回回答内容，改善用户体验
    """
    from fastapi.responses import StreamingResponse

    async def generate():
        try:
            assistant = get_fund_qa_assistant()
            answer = await asyncio.to_thread(assistant.process_query, query)
            # 模拟流式输出：按段落逐步发送
            paragraphs = answer.split('\n\n')
            for i, para in enumerate(paragraphs):
                chunk = para if i == 0 else '\n\n' + para
                yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.05)
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error("流式问答异常: %s", str(e))
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/v1/wealth-advisor/stream", tags=["财富顾问（流式）"])
async def wealth_advisor_stream(
    query: str,
    customer_id: str = "customer1",
    thread_id: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    财富顾问流式接口（SSE）
    逐步返回响应内容
    """
    from fastapi.responses import StreamingResponse

    async def generate():
        try:
            from hybrid_wealth_advisor_langgraph import run_wealth_advisor
            # 先发送处理模式提示
            yield f"data: {json.dumps({'status': 'processing', 'message': '正在分析您的问题...'}, ensure_ascii=False)}\n\n"

            result = await asyncio.wait_for(
                asyncio.to_thread(run_wealth_advisor, query, customer_id, thread_id),
                timeout=settings.request_timeout
            )

            mode = result.get("processing_mode", "未知")
            yield f"data: {json.dumps({'status': 'mode', 'mode': mode}, ensure_ascii=False)}\n\n"

            response_text = result.get("final_response", "未生成响应")
            if result.get("error"):
                response_text = f"处理出错：{result['error']}"

            # 按段落流式输出
            paragraphs = response_text.split('\n\n')
            for i, para in enumerate(paragraphs):
                chunk = para if i == 0 else '\n\n' + para
                yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.05)

            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'error': '处理超时，请稍后重试'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error("流式财富顾问异常: %s", str(e))
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/v1/research/stream", tags=["投研分析（流式）"])
async def research_stream(
    topic: str,
    industry: str = "综合",
    horizon: str = "中期",
    thread_id: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    投研分析流式接口（SSE）
    实时推送五阶段（感知→建模→推理→决策→报告）的进度和结果
    支持多轮对话：传入 thread_id 可保持上下文记忆
    """
    from fastapi.responses import StreamingResponse
    from deliberative_research_langgraph import run_research_agent_stream

    async def generate():
        report_text = ""
        try:
            # 多轮对话上下文
            context_messages = []
            if thread_id:
                try:
                    history = db.get_conversation_history(thread_id)
                    context_messages = [(m["role"], m["content"]) for m in history[-4:]]
                except Exception as e:
                    logger.debug("获取投研流式上下文失败: %s", str(e))

            if context_messages:
                context_str = "\n".join([f"[{r}]: {c[:200]}" for r, c in context_messages])
                full_topic = f"此前对话：\n{context_str}\n\n当前研究主题：{topic}"
            else:
                full_topic = topic

            def sync_gen():
                return run_research_agent_stream(full_topic, industry, horizon)

            import concurrent.futures
            loop = asyncio.get_event_loop()
            gen = await loop.run_in_executor(None, sync_gen)

            # 将迭代移到线程中，通过队列传递
            queue = asyncio.Queue()
            stop_event = threading.Event()

            def producer():
                try:
                    for event in gen():
                        if stop_event.is_set():
                            break
                        asyncio.run_coroutine_threadsafe(queue.put(("data", event)), loop)
                    asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)

            thread = threading.Thread(target=producer, daemon=True)
            thread.start()

            while True:
                try:
                    msg_type, data = await asyncio.wait_for(queue.get(), timeout=settings.request_timeout)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'event': 'error', 'error': '处理超时，请稍后重试'}, ensure_ascii=False)}\n\n"
                    stop_event.set()
                    return

                if msg_type == "done":
                    # 持久化对话记录
                    try:
                        tid = thread_id or f"research-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        db.save_conversation(thread_id=tid, role="user", content=topic, tab_type="research")
                        db.save_conversation(thread_id=tid, role="assistant", content=report_text[:500], tab_type="research")
                    except Exception as ps_err:
                        logger.debug("保存流式对话记录失败: %s", str(ps_err))
                    return
                elif msg_type == "error":
                    yield f"data: {json.dumps({'error': data}, ensure_ascii=False)}\n\n"
                    return
                else:
                    # 累积报告内容（用于持久化）
                    if isinstance(data, dict) and data.get("event") == "complete":
                        report_text = data.get("report", "")
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)

        except Exception as e:
            logger.error("流式投研异常: %s", str(e))
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ========== 系统监控端点 ==========

class SystemMetrics(BaseModel):
    """系统监控指标"""
    uptime_seconds: float
    total_requests: int
    rate_limited_count: int
    db_size_mb: float
    api_status: str
    kb_status: str


# 全局计数器（修复：添加线程锁保证并发安全）
_start_time = time.time()
_total_requests = 0
_rate_limited_count = 0
_counters_lock = threading.Lock()


@app.get("/api/v1/health/metrics", response_model=SystemMetrics, tags=["系统监控"])
async def system_metrics():
    """
    系统监控指标端点
    返回运行状态、请求统计、数据库大小等信息
    """
    global _total_requests, _rate_limited_count
    db_size = os.path.getsize(str(db.DB_PATH)) / (1024 * 1024) if os.path.exists(str(db.DB_PATH)) else 0

    # 检查知识库状态
    kb_status = "未加载"
    try:
        from knowledge_base import get_knowledge_base
        kb = get_knowledge_base()
        stats = kb.get_statistics()
        kb_status = f"已加载 ({stats['total_documents']}条文档, {'向量' if stats['embedding_enabled'] else '文本'}模式)"
    except Exception:
        pass

    return SystemMetrics(
        uptime_seconds=round(time.time() - _start_time, 1),
        total_requests=_total_requests,
        rate_limited_count=_rate_limited_count,
        db_size_mb=round(db_size, 2),
        api_status="healthy",
        kb_status=kb_status,
    )


@app.get("/api/v1/health/detailed", tags=["系统监控"])
async def detailed_health():
    """
    详细健康检查（含LLM连接测试、数据库检查、知识库状态）
    """
    health_info = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "database": "ok" if os.path.exists(str(db.DB_PATH)) else "error",
            "llm": "unknown",
            "knowledge_base": "unknown",
        },
        "version": settings.app_version,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }

    # 检查LLM
    from llm_factory import is_llm_available
    health_info["checks"]["llm"] = "available" if is_llm_available() else "unavailable"

    # 检查知识库
    try:
        from knowledge_base import get_knowledge_base
        kb = get_knowledge_base()
        health_info["checks"]["knowledge_base"] = f"loaded ({len(kb.documents)} docs)"
    except Exception as e:
        health_info["checks"]["knowledge_base"] = f"error: {str(e)}"

    # 如有任一检查失败，设置整体状态为 degraded
    if any(v == "error" or v.startswith("error") for v in health_info["checks"].values()):
        health_info["status"] = "degraded"

    return health_info


# 更新中间件以增加请求计数
@app.middleware("http")
async def enhanced_request_middleware(request: Request, call_next):
    """
    增强版请求中间件：速率限制 + 请求计数 + 日志（线程安全）
    """
    global _total_requests, _rate_limited_count
    with _counters_lock:
        _total_requests += 1

    client_ip = request.client.host if request.client else "unknown"
    start_time = time.time()

    # 速率限制检查（健康检查接口和根路径豁免）
    skip_paths = {"/", "/docs", "/redoc", "/openapi.json"}
    if request.url.path not in skip_paths and not request.url.path.startswith("/api/v1/health"):
        if not rate_limiter.is_allowed(client_ip):
            with _counters_lock:
                _rate_limited_count += 1
            logger.warning("速率限制触发: IP=%s, Path=%s", client_ip, request.url.path)
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": f"请求过于频繁，请稍后再试（限制：{settings.rate_limit_per_minute}次/分钟）"}
            )

    # 执行请求
    try:
        response = await call_next(request)
    except asyncio.TimeoutError:
        logger.error("请求超时: IP=%s, Path=%s", client_ip, request.url.path)
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=504, content={"detail": "请求处理超时"})
    except Exception as e:
        logger.error("请求处理异常: IP=%s, Path=%s, Error=%s", client_ip, request.url.path, str(e))
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"detail": "服务器内部错误"})

    # 请求日志
    process_time = time.time() - start_time
    logger.info(
        "请求: %s %s IP=%s 耗时=%.2fs 状态=%d",
        request.method, request.url.path, client_ip, process_time, response.status_code
    )

    return response


# ========== 启动入口 ==========

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print(f"  {settings.app_name} - Web API服务")
    print(f"  API文档: http://localhost:{settings.app_port}/docs")
    print(f"  健康检查: http://localhost:{settings.app_port}/")
    print(f"  API认证: {'已启用' if _get_valid_api_keys() else '未启用'}")
    print(f"  速率限制: {settings.rate_limit_per_minute}次/分钟")
    print(f"  日志文件: {settings.log_file}")
    print("=" * 60)
    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
        log_level="info"
    )
