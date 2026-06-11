"""
pytest全局配置和共享fixtures
"""
import os
import sys
import pytest

# 将项目根目录和各子目录添加到模块搜索路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "01-私募基金运作指引问答助手（反应式）"))
sys.path.insert(0, os.path.join(BASE_DIR, "02-智能投研助手（深思熟虑）"))
sys.path.insert(0, os.path.join(BASE_DIR, "03-投顾AI助手（混合式）"))


@pytest.fixture
def mock_dashscope_api_key(monkeypatch):
    """
    设置模拟的DASHSCOPE_API_KEY环境变量
    :param monkeypatch: pytest内置fixture
    """
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-mock-api-key-123456")


@pytest.fixture
def fund_qa_assistant(mock_dashscope_api_key):
    """
    获取私募基金问答助手实例
    :param mock_dashscope_api_key: 模拟API Key fixture
    :return: FundQAAssistant实例
    """
    from fund_qa_langgraph_v2 import FundQAAssistant
    return FundQAAssistant()
