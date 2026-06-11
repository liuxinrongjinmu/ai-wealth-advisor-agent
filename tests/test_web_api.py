"""
Web API接口测试
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(mock_dashscope_api_key):
    """
    获取FastAPI测试客户端
    :param mock_dashscope_api_key: 模拟API Key fixture
    :return: TestClient实例
    """
    # 需要在导入web_api前设置环境变量
    from web_api import app
    return TestClient(app)


class TestHealthCheck:
    """健康检查接口测试"""

    def test_health_check(self, client):
        """
        测试健康检查接口返回正确状态
        """
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "version" in data
        assert "auth_enabled" in data


class TestFundQAEndpoint:
    """私募基金问答接口测试"""

    def test_fund_qa_missing_query(self, client):
        """
        测试缺少query参数时返回422
        """
        response = client.post("/api/v1/fund-qa", json={})
        assert response.status_code == 422

    def test_fund_qa_empty_query(self, client):
        """
        测试空query时返回422
        """
        response = client.post("/api/v1/fund-qa", json={"query": ""})
        assert response.status_code == 422

    def test_fund_qa_too_long_query(self, client):
        """
        测试超长query时返回422
        """
        response = client.post("/api/v1/fund-qa", json={"query": "x" * 501})
        assert response.status_code == 422


class TestResearchEndpoint:
    """投研分析接口测试"""

    def test_research_missing_topic(self, client):
        """
        测试缺少topic参数时返回422
        """
        response = client.post("/api/v1/research", json={})
        assert response.status_code == 422

    def test_research_default_params(self, client):
        """
        测试投研分析默认参数
        """
        response = client.post("/api/v1/research", json={"topic": "新能源"})
        # 可能因API Key无效而500，但不应是422参数错误
        assert response.status_code != 422


class TestWealthAdvisorEndpoint:
    """财富顾问接口测试"""

    def test_wealth_advisor_missing_query(self, client):
        """
        测试缺少query参数时返回422
        """
        response = client.post("/api/v1/wealth-advisor", json={})
        assert response.status_code == 422

    def test_wealth_advisor_default_customer(self, client):
        """
        测试财富顾问默认客户ID
        """
        response = client.post("/api/v1/wealth-advisor", json={"query": "测试问题"})
        # 不应是参数校验错误
        assert response.status_code != 422


class TestRateLimiting:
    """速率限制测试"""

    def test_rate_limit_headers(self, client):
        """
        测试请求正常通过（未超限）
        """
        response = client.get("/")
        assert response.status_code == 200


class TestLegacyEndpoints:
    """兼容旧版API路径测试"""

    def test_legacy_fund_qa_exists(self, client):
        """
        测试旧版基金问答路径存在
        """
        response = client.post("/api/fund-qa", json={"query": "测试"})
        # 不应返回404
        assert response.status_code != 404

    def test_legacy_research_exists(self, client):
        """
        测试旧版投研分析路径存在
        """
        response = client.post("/api/research", json={"topic": "测试"})
        assert response.status_code != 404

    def test_legacy_wealth_advisor_exists(self, client):
        """
        测试旧版财富顾问路径存在
        """
        response = client.post("/api/wealth-advisor", json={"query": "测试"})
        assert response.status_code != 404
