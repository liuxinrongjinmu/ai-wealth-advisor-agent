"""
共享LLM工厂模块
统一管理通义千问LLM实例的创建和配置，消除三个子系统中的代码重复。
"""
import os
import logging
from typing import Optional
from langchain_community.llms import Tongyi

logger = logging.getLogger(__name__)

_llm_instance: Optional[Tongyi] = None


def get_llm(model_name: str = "Qwen-Turbo-2025-04-28") -> Tongyi:
    """
    获取LLM实例（全局单例，延迟初始化）
    :param model_name: 模型名称，默认 Qwen-Turbo-2025-04-28
    :return: Tongyi LLM实例
    :raises ValueError: 未配置DASHSCOPE_API_KEY时抛出
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance
    api_key = os.getenv('DASHSCOPE_API_KEY', '')
    if not api_key:
        raise ValueError("未检测到DASHSCOPE_API_KEY环境变量，请先设置：set DASHSCOPE_API_KEY=your-api-key")
    _llm_instance = Tongyi(model_name=model_name, dashscope_api_key=api_key)
    logger.info("LLM实例初始化完成: model=%s", model_name)
    return _llm_instance


def reset_llm():
    """
    重置LLM实例（主要用于测试环境切换模型）
    """
    global _llm_instance
    _llm_instance = None
    logger.info("LLM实例已重置")


def is_llm_available() -> bool:
    """
    检查LLM是否可用（API Key是否已配置）
    :return: True表示已配置
    """
    api_key = os.getenv('DASHSCOPE_API_KEY', '')
    return bool(api_key)