"""
RAG知识库模块 - 基于向量检索的法规知识增强

为私募基金问答助手提供语义检索能力，减少对LLM的裸依赖，
提升回答的专业性和可信度。

核心功能：
1. 将22条规则文档向量化存储
2. 支持语义相似度检索
3. 与关键词匹配互补，提供更精准的答案

依赖：需要安装 sentence-transformers 和 chromadb（可选）
基础模式使用 difflib 进行文本相似度匹配，无需额外依赖
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 知识库存储路径
KB_DIR = Path(__file__).parent / "knowledge_base"
KB_DIR.mkdir(exist_ok=True)


class KnowledgeBase:
    """
    法规知识库
    支持语义检索和关键词匹配双模式
    """

    def __init__(self):
        """初始化知识库，加载规则文档"""
        self.documents: List[Dict[str, Any]] = []
        self._load_documents()
        self._embeddings_cache: Dict[str, List[float]] = {}
        self._embedding_model = None
        self._try_load_embeddings()

    def _load_documents(self):
        """
        加载22条规则文档到知识库
        数据来源与 fund_qa_langgraph_v2.py 保持一致
        """
        self.documents = [
            {
                "id": "rule001",
                "category": "设立与募集",
                "title": "私募基金的合格投资者标准",
                "content": "合格投资者是指具备相应风险识别能力和风险承担能力，投资于单只私募基金的金额不低于100万元且符合下列条件之一的单位和个人：1. 净资产不低于1000万元的单位 2. 金融资产不低于300万元或者最近三年个人年均收入不低于50万元的个人",
                "keywords": ["合格投资者", "100万元", "1000万元", "300万元", "50万元", "净资产", "金融资产"]
            },
            {
                "id": "rule002",
                "category": "设立与募集",
                "title": "私募基金最低募集规模要求",
                "content": "私募证券投资基金的最低募集规模不得低于人民币1000万元。对于私募股权基金、创业投资基金等其他类型的私募基金，监管规定更加灵活，通常需符合基金合同的约定。",
                "keywords": ["最低募集规模", "1000万元", "募集规模", "成立条件"]
            },
            {
                "id": "rule003",
                "category": "设立与募集",
                "title": "私募基金管理人资质要求",
                "content": "私募基金管理人需要在中国证券投资基金业协会登记。登记前提条件包括：1. 公司已依法成立并运营满两年 2. 高管人员具备从业资格 3. 最近两年经审计的财务报告 4. 建立完善的风险管理体系。登记后获得私募基金管理人牌照。",
                "keywords": ["管理人资质", "登记", "协会", "从业资格", "财务报告", "风险管理体系", "牌照"]
            },
            {
                "id": "rule004",
                "category": "设立与募集",
                "title": "私募基金募集期规定",
                "content": "私募基金的募集期通常为6个月。在募集期内，管理人需要从合格投资者中募集足够的资金。募集期满后应当在20个工作日内向协会进行基金备案。",
                "keywords": ["募集期", "6个月", "备案", "20个工作日", "募集时间", "募集"]
            },
            {
                "id": "rule005",
                "category": "监管规定",
                "title": "风险准备金要求",
                "content": "私募证券基金管理人应当按照管理费收入的10%计提风险准备金，主要用于赔偿因管理人违法违规、违反基金合同、操作错误等给基金财产或者投资者造成的损失。",
                "keywords": ["风险准备金", "10%", "管理费收入", "赔偿"]
            },
            {
                "id": "rule006",
                "category": "监管规定",
                "title": "私募基金风险等级划分",
                "content": "私募基金按风险程度分为五个等级：R1（谨慎型）- 风险最低，主要投资债券和现金；R2（稳健型）- 风险较低，混合投资；R3（平衡型）- 风险中等，股债混合；R4（积极型）- 风险较高，主要投资股票；R5（激进型）- 风险最高，可投资衍生品等高风险资产。",
                "keywords": ["风险等级", "R1", "R2", "R3", "R4", "R5", "谨慎型", "稳健型", "平衡型", "积极型", "激进型"]
            },
            {
                "id": "rule007",
                "category": "监管规定",
                "title": "私募基金管理人责任",
                "content": "私募基金管理人的主要责任包括：1. 忠实义务 - 恪尽职守，维护投资者利益 2. 勤勉义务 - 勤勉尽职，制定科学的投资策略 3. 披露义务 - 及时真实披露基金信息 4. 风控责任 - 建立有效的风险管理体系 5. 信息保管 - 保护投资者的个人信息",
                "keywords": ["管理人责任", "忠实义务", "勤勉义务", "披露义务", "风控责任", "信息保管"]
            },
            {
                "id": "rule008",
                "category": "信息披露",
                "title": "投资者信息披露要求",
                "content": "私募基金应当定期向投资者披露以下信息：1. 基金净值及单位净值 2. 投资运作情况 3. 主要财务指标 4. 基金经理变更等重大事项 5. 风险提示信息。定期披露通常每季度进行一次（书面形式）或每月进行一次（电子形式）。",
                "keywords": ["信息披露", "基金净值", "投资运作", "财务指标", "基金经理变更", "重大事项", "风险提示"]
            },
            {
                "id": "rule009",
                "category": "信息披露",
                "title": "基金合同必备内容",
                "content": "基金合同是管理人与投资者的权利义务协议，必须包含：1. 当事人的权利和义务 2. 基金资产的管理、运用、处分原则 3. 基金资产估值方法 4. 收益分配原则 5. 基金的存续期及终止事由 6. 相关费用的计算和支付 7. 信息披露方式 8. 风险揭示等内容",
                "keywords": ["基金合同", "合同内容", "权利义务", "资产管理", "估值方法", "收益分配", "存续期", "终止事由", "风险揭示"]
            },
            {
                "id": "rule010",
                "category": "信息披露",
                "title": "向监管部门的报告义务",
                "content": "私募基金管理人需要向中国基金业协会进行以下报告：1. 定期报告 - 至少每季度提交一次 2. 重大事项报告 - 15个工作日内报告基金经理变更、重大违约、重大诉讼等 3. 临时报告 - 对基金产生重大影响的事项 4. 年度报告 - 包含基金业绩、投资情况等详细信息",
                "keywords": ["监管部门报告", "定期报告", "重大事项报告", "临时报告", "年度报告", "基金业绩"]
            },
            {
                "id": "rule011",
                "category": "投资范围",
                "title": "私募基金投资范围",
                "content": "私募基金的投资范围包括：1. 上市公司股票和非上市企业股权 2. 债券、票据等固定收益资产 3. 商品、金融衍生品等 4. 不动产及其他资产 5. 法律法规允许的其他资产。具体的投资范围由基金合同约定。",
                "keywords": ["投资范围", "股票", "股权", "债券", "衍生品", "不动产"]
            },
            {
                "id": "rule012",
                "category": "投资范围",
                "title": "投资集中度限制",
                "content": "为防范风险，私募基金的投资集中度通常受限：1. 对单个企业的投资不得超过基金资产总值的20% 2. 对同一类资产的投资比例受基金合同约束 3. 与管理人存在关联关系的投资有严格限制 4. 不得进行法律禁止的投资活动。这些限制在基金合同中有具体规定。",
                "keywords": ["投资集中度", "集中度限制", "单个企业", "资产总值", "20%", "关联关系"]
            },
            {
                "id": "rule013",
                "category": "费用结构",
                "title": "私募基金费用结构",
                "content": "私募基金的主要费用包括：1. 管理费 - 按基金资产净值的一定比例计提（通常1%-2%） 2. 业绩报酬 - 按超额收益的一定比例计提（通常20%） 3. 保管费 - 由资产保管人收取（通常0.1%-0.25%） 4. 其他费用 - 审计费、律师费、信息披露费等。所有费用应在基金合同中明确列示。",
                "keywords": ["费用", "管理费", "业绩报酬", "保管费", "审计费", "律师费"]
            },
            {
                "id": "rule014",
                "category": "费用结构",
                "title": "管理费计算方式",
                "content": "管理费是私募基金管理人因管理基金而获得的报酬。计算方法：1. 按基金资产净值的年度百分比计提 2. 通常范围为基金净值的0.5%-3% 3. 每月计提，按年支付或每年支付一次 4. 按实际计提日期与当年天数的比例计算。例如：基金净值1亿元，年费率2%，则年管理费为200万元。",
                "keywords": ["管理费", "计算", "基金资产净值", "年度百分比", "0.5%-3%", "每月计提", "按年支付"]
            },
            {
                "id": "rule015",
                "category": "费用结构",
                "title": "业绩报酬计提规则",
                "content": "业绩报酬是管理人获得的超额收益分成。特点：1. 仅当基金产生正收益时才计提 2. 通常按超额收益（高于基准收益率的部分）的20%-30%计提 3. 有些基金采用高水位线机制，确保投资者不重复支付 4. 计提方式和条件在基金合同中明确规定 5. 计提时间通常为年度或基金清算时。",
                "keywords": ["业绩报酬", "超额收益", "正收益", "20%-30%", "高水位线", "基准收益率", "计提条件"]
            },
            {
                "id": "rule016",
                "category": "退出与清算",
                "title": "投资者退出方式",
                "content": "投资者的退出方式包括：1. 基金清算 - 在基金终止时获得清算收益 2. 二级市场转让 - 向其他合格投资者转让份额 3. 管理人回购 - 向基金管理人申请回购 4. 权益转让 - 将基金权益转让给其他机构 5. 正常赎回 - 在开放期内赎回。具体退出方式由基金合同约定。",
                "keywords": ["退出", "清算", "二级市场转让", "管理人回购", "权益转让", "正常赎回"]
            },
            {
                "id": "rule017",
                "category": "退出与清算",
                "title": "基金清算资产分配顺序",
                "content": "基金清算时的资产分配顺序为：1. 支付基金清算费用 2. 支付基金债务（包括基金保管费、审计费等） 3. 支付管理人的管理费（如合同要求） 4. 向投资者返还基金份额对应的净资产。清算应在基金终止后的规定时间内完成，并将清算报告报送协会。",
                "keywords": ["清算", "资产分配", "清算费用", "基金债务", "保管费", "审计费", "管理费", "净资产", "清算报告"]
            },
            {
                "id": "rule018",
                "category": "退出与清算",
                "title": "强制清算触发条件",
                "content": "以下情况可能导致私募基金被强制清算：1. 基金合同约定的终止事由出现 2. 基金资产净值持续低于合同约定的最低规模 3. 管理人被撤销牌照或发生严重违规 4. 投资者大会或法律规定的其他情形 5. 基金资产发生重大损失。强制清算需按法律程序进行，保护投资者合法权益。",
                "keywords": ["强制清算", "终止事由", "资产净值", "最低规模", "撤销牌照", "严重违规", "投资者大会", "重大损失"]
            },
            {
                "id": "rule019",
                "category": "风险管理",
                "title": "私募基金主要风险",
                "content": "私募基金面临的主要风险包括：1. 市场风险 - 投资品种价格波动 2. 流动性风险 - 基金资产难以变现 3. 信用风险 - 债务人违约风险 4. 管理风险 - 管理人操作失误或道德风险 5. 政策风险 - 法律法规变化 6. 集中度风险 - 投资过于集中。投资者需充分了解这些风险。",
                "keywords": ["风险", "市场风险", "流动性风险", "信用风险", "管理风险", "政策风险", "集中度风险"]
            },
            {
                "id": "rule020",
                "category": "风险管理",
                "title": "风险管理措施",
                "content": "私募基金管理人的风险管理措施包括：1. 建立完善的风险管理制度和流程 2. 设置独立的风险管理部门 3. 定期进行风险评估和压力测试 4. 建立预警机制和应急预案 5. 对投资组合进行监测和调整 6. 计提风险准备金 7. 进行专业人员培训 8. 接受外部审计和监督。",
                "keywords": ["风险管理", "风险防控", "风险管理制度", "风险管理部门", "风险评估", "压力测试", "预警机制", "应急预案", "风险准备金", "人员培训", "外部审计"]
            },
            {
                "id": "rule021",
                "category": "合规要求",
                "title": "私募基金管理人合规要求",
                "content": "私募基金管理人的合规要求包括：1. 建立合规风控部门，配备合规人员 2. 制定合规管理制度和业务规则 3. 防范利益冲突，进行关联交易管理 4. 确保投资者风险评估适当 5. 防范洗钱和恐怖融资 6. 保护客户隐私和信息安全 7. 接受协会自律管理和监管 8. 定期更新公司治理和内部管理。",
                "keywords": ["合规要求", "合规义务", "风控部门", "合规人员", "管理制度", "利益冲突", "关联交易", "洗钱", "恐怖融资", "客户隐私", "信息安全", "自律管理", "公司治理"]
            },
            {
                "id": "rule022",
                "category": "合规要求",
                "title": "私募基金禁止行为",
                "content": "私募基金禁止的行为包括：1. 向不合格投资者募集 2. 挪用基金资产 3. 承诺保本保收益 4. 虚假宣传、误导性宣传 5. 不公平对待投资者 6. 从事内幕交易、操纵市场、违法关联交易等。上述行为均属于违法违规。",
                "keywords": ["禁止行为", "内幕交易", "操纵市场", "挪用基金资产", "承诺保本", "虚假宣传"]
            },
        ]
        logger.info("知识库加载完成，共 %d 条规则文档", len(self.documents))

    def _try_load_embeddings(self):
        """
        尝试加载嵌入模型（可选依赖）
        如果 sentence-transformers 不可用，使用基于文本的相似度匹配
        """
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("嵌入模型加载成功: paraphrase-multilingual-MiniLM-L12-v2")
            # 预计算所有文档的嵌入向量
            for doc in self.documents:
                embedding = self._embedding_model.encode(doc['content'])
                self._embeddings_cache[doc['id']] = embedding.tolist()
        except ImportError:
            logger.info("sentence-transformers 未安装，使用基于文本的相似度匹配")
        except Exception as e:
            logger.warning("嵌入模型加载失败: %s，降级为文本匹配", str(e))

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        搜索最相关的知识文档
        :param query: 用户查询文本
        :param top_k: 返回结果数量
        :return: 相关文档列表，按相似度降序排列
        """
        if self._embedding_model is not None:
            return self._semantic_search(query, top_k)
        return self._text_search(query, top_k)

    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        基于语义嵌入的搜索
        :param query: 用户查询
        :param top_k: 返回数量
        :return: 相关文档列表
        """
        import numpy as np
        query_embedding = self._embedding_model.encode(query)

        results = []
        for doc in self.documents:
            doc_embedding = np.array(self._embeddings_cache.get(doc['id'], []))
            if len(doc_embedding) == 0:
                continue
            # 余弦相似度
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            results.append({
                **doc,
                'score': float(similarity),
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def _text_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        基于文本关键词和相似度的搜索（降级方案）
        :param query: 用户查询
        :param top_k: 返回数量
        :return: 相关文档列表
        """
        from difflib import SequenceMatcher

        results = []
        query_lower = query.lower()

        for doc in self.documents:
            # 关键词匹配得分
            kw_score = 0
            keywords = doc.get('keywords', [])
            for kw in keywords:
                if kw.lower() in query_lower:
                    kw_score += 10  # 每个匹配关键词10分

            # 文本相似度得分
            text_sim = SequenceMatcher(
                None, query_lower, doc['content'].lower()
            ).ratio()

            # 综合得分：关键词70% + 语义30%
            combined = kw_score * 0.7 + text_sim * 100 * 0.3

            results.append({
                **doc,
                'score': combined,
                'kw_score': kw_score,
                'text_sim': text_sim,
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        获取所有知识文档
        :return: 文档列表
        """
        return self.documents

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取文档
        :param doc_id: 文档ID
        :return: 文档或None
        """
        for doc in self.documents:
            if doc['id'] == doc_id:
                return doc
        return None

    def add_document(self, doc_id: str, category: str, title: str, content: str, keywords: List[str] = None):
        """
        动态添加知识文档
        :param doc_id: 文档ID
        :param category: 分类
        :param title: 标题
        :param content: 内容
        :param keywords: 关键词列表
        """
        new_doc = {
            "id": doc_id,
            "category": category,
            "title": title,
            "content": content,
            "keywords": keywords or [],
        }
        self.documents.append(new_doc)
        logger.info("知识库新增文档: %s - %s", doc_id, title)

        # 如果嵌入模型可用，计算新文档的嵌入向量
        if self._embedding_model is not None:
            embedding = self._embedding_model.encode(content)
            self._embeddings_cache[doc_id] = embedding.tolist()

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取知识库统计信息
        :return: 统计信息字典
        """
        categories = {}
        for doc in self.documents:
            cat = doc['category']
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_documents": len(self.documents),
            "categories": categories,
            "embedding_enabled": self._embedding_model is not None,
            "embedding_model": str(self._embedding_model) if self._embedding_model else "text-based",
        }


# 全局单例
_kb_instance: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """
    获取知识库单例
    :return: KnowledgeBase 实例
    """
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance


# 测试入口
if __name__ == "__main__":
    kb = get_knowledge_base()
    print(f"知识库统计: {kb.get_statistics()}")

    # 测试搜索
    test_queries = [
        "合格投资者需要什么条件？",
        "私募基金有哪些风险？",
        "管理费怎么计算？",
        "可以投资股票吗？",
        "如何退出私募基金？",
    ]

    for q in test_queries:
        print(f"\n查询: {q}")
        results = kb.search(q, top_k=2)
        for r in results:
            print(f"  [{r['category']}] {r['title']} (得分: {r['score']:.2f})")