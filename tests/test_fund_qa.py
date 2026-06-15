"""
私募基金问答助手单元测试
"""
import pytest


class TestFundQAAssistant:
    """私募基金问答助手测试类"""

    def test_initialize_rules_db(self, fund_qa_assistant):
        """
        测试规则数据库初始化
        """
        rules = fund_qa_assistant.rules_db
        assert len(rules) == 22, f"规则数量应为22，实际为{len(rules)}"

    def test_keyword_weights_initialized(self, fund_qa_assistant):
        """
        测试关键词权重初始化
        """
        weights = fund_qa_assistant.keyword_weights
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_calculate_keyword_scores(self, fund_qa_assistant):
        """
        测试关键词评分计算
        """
        scores = fund_qa_assistant._calculate_keyword_scores("合格投资者标准是什么？")
        assert isinstance(scores, dict)
        assert len(scores) > 0

    def test_calculate_semantic_scores(self, fund_qa_assistant):
        """
        测试语义相似度评分计算
        """
        scores = fund_qa_assistant._calculate_semantic_scores("合格投资者标准是什么？")
        assert isinstance(scores, dict)
        assert len(scores) > 0

    def test_combine_scores(self, fund_qa_assistant):
        """
        测试综合评分计算
        """
        keyword_scores = fund_qa_assistant._calculate_keyword_scores("合格投资者标准")
        semantic_scores = fund_qa_assistant._calculate_semantic_scores("合格投资者标准")
        combined = fund_qa_assistant._combine_scores(keyword_scores, semantic_scores)
        assert isinstance(combined, dict)
        # 综合评分应介于0-1之间
        for score in combined.values():
            assert 0 <= score <= 1

    def test_select_best_match(self, fund_qa_assistant):
        """
        测试最佳匹配选择
        """
        scores = {"rule_01": 0.9, "rule_02": 0.5, "rule_03": 0.3}
        best_id, best_score = fund_qa_assistant._select_best_match(scores)
        assert best_id == "rule_01"
        assert best_score == 0.9

    def test_get_rule_by_id(self, fund_qa_assistant):
        """
        测试按ID获取规则
        """
        rule = fund_qa_assistant._get_rule_by_id("rule001")
        assert rule is not None
        assert "question" in rule
        assert "answer" in rule

    def test_get_rule_by_id_not_found(self, fund_qa_assistant):
        """
        测试获取不存在的规则
        """
        rule = fund_qa_assistant._get_rule_by_id("rule_nonexistent")
        assert rule is None

    def test_cache_functionality(self, fund_qa_assistant):
        """
        测试查询缓存功能
        """
        assert isinstance(fund_qa_assistant._query_cache, dict)
        assert fund_qa_assistant._cache_size == 128

    def test_classify_query_qualified_investor(self, fund_qa_assistant):
        """
        测试合格投资者查询分类
        """
        query = "合格投资者标准是什么？"
        keyword_scores = fund_qa_assistant._calculate_keyword_scores(query)
        assert len(keyword_scores) > 0, "合格投资者查询应有匹配结果"

    def test_classify_query_fund_fees(self, fund_qa_assistant):
        """
        测试基金费用查询分类
        """
        query = "私募基金有哪些费用？"
        keyword_scores = fund_qa_assistant._calculate_keyword_scores(query)
        assert len(keyword_scores) > 0, "基金费用查询应有匹配结果"


class TestFundQAEdgeCases:
    """私募基金问答边界条件测试"""

    def test_empty_query_scores(self, fund_qa_assistant):
        """
        测试空查询评分
        """
        scores = fund_qa_assistant._calculate_keyword_scores("")
        assert isinstance(scores, dict)

    def test_very_long_query(self, fund_qa_assistant):
        """
        测试超长查询
        """
        long_query = "合格投资者" * 100
        scores = fund_qa_assistant._calculate_keyword_scores(long_query)
        assert isinstance(scores, dict)

    def test_special_characters_query(self, fund_qa_assistant):
        """
        测试特殊字符查询
        """
        query = "!@#$%^&*()合格投资者"
        scores = fund_qa_assistant._calculate_keyword_scores(query)
        assert isinstance(scores, dict)
