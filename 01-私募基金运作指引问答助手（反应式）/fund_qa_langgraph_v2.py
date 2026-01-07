#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
私募基金运作指引问答助手（反应式）

本系统是基于LangChain和大语言模型（LLM）的私募基金智能问答机器人，
专为私募基金行业的法规、合规、投资、费用、风险等22条核心规则提供自动化、智能化的问答服务。

1. 内置22条权威私募基金规则，支持标准问法、关键词、口语化、场景化等多种提问方式。
2. 多级智能匹配：特殊处理器优先、关键词权重自动加权、语义理解（LLM补全）。
3. 支持命令行交互、批量测试、代码集成，适合金融从业者、合规人员、投资者及AI开发者。
4. 易扩展：规则库、关键词权重、特殊模式均可自定义扩展。
"""

import re
import json
from typing import Dict, List, Any, Tuple, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Tongyi
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from difflib import SequenceMatcher

# 设置API密钥
DASHSCOPE_API_KEY = 'sk-882e296067b744289acf27e6e20f3ec0'

# 创建LLM实例
llm = Tongyi(model_name="Qwen-Turbo-2025-04-28", dashscope_api_key=DASHSCOPE_API_KEY)


class FundQAAssistant:
    """私募基金问答助手核心类"""

    def __init__(self):
        """初始化助手"""
        self.llm = llm
        self.rules_db = self._initialize_rules_db()
        self.keyword_weights = self._initialize_keyword_weights()
        self.special_handlers = self._initialize_special_handlers()

    def _initialize_rules_db(self) -> List[Dict[str, Any]]:
        """初始化22条私募基金规则数据库"""
        return [
            # ========== 设立与募集 ==========
            {
                "id": "rule001",
                "category": "设立与募集",
                "question": "私募基金的合格投资者标准是什么？",
                "answer": "合格投资者是指具备相应风险识别能力和风险承担能力，投资于单只私募基金的金额不低于100万元且符合下列条件之一的单位和个人：\n1. 净资产不低于1000万元的单位\n2. 金融资产不低于300万元或者最近三年个人年均收入不低于50万元的个人",
                "keywords": ["合格投资者", "100万元", "1000万元", "300万元", "50万元", "净资产", "金融资产", "年均收入", "最低金额", "投资条件", "投资者标准", "投资私募"]
            },
            {
                "id": "rule002",
                "category": "设立与募集",
                "question": "私募基金的最低募集规模要求是多少？",
                "answer": "私募证券投资基金的最低募集规模不得低于人民币1000万元。对于私募股权基金、创业投资基金等其他类型的私募基金，监管规定更加灵活，通常需符合基金合同的约定。",
                "keywords": ["最低募集规模", "1000万元", "募集规模", "成立条件", "最低规模", "募集", "多少钱", "成立"]
            },
            {
                "id": "rule003",
                "category": "设立与募集",
                "question": "私募基金管理人需要什么资质？",
                "answer": "私募基金管理人需要在中国证券投资基金业协会登记。登记前提条件包括：\n1. 公司已依法成立并运营满两年\n2. 高管人员具备从业资格\n3. 最近两年经审计的财务报告\n4. 建立完善的风险管理体系\n登记后获得私募基金管理人牌照",
                "keywords": ["管理人资质", "登记", "协会", "从业资格", "财务报告", "风险管理体系", "牌照", "管理人", "成为", "资质"]
            },
            {
                "id": "rule004",
                "category": "设立与募集",
                "question": "私募基金募集期一般是多长时间？",
                "answer": "私募基金的募集期通常为6个月。在募集期内，管理人需要从合格投资者中募集足够的资金。募集期满后应当在20个工作日内向协会进行基金备案。",
                "keywords": ["募集期", "6个月", "备案", "20个工作日", "募集时间", "募集", "多长", "花多久"]
            },

            # ========== 监管规定 ==========
            {
                "id": "rule005",
                "category": "监管规定",
                "question": "私募基金管理人的风险准备金要求是什么？",
                "answer": "私募证券基金管理人应当按照管理费收入的10%计提风险准备金，主要用于赔偿因管理人违法违规、违反基金合同、操作错误等给基金财产或者投资者造成的损失。",
                "keywords": ["风险准备金", "10%", "管理费收入", "赔偿", "违法违规", "操作错误"]
            },
            {
                "id": "rule006",
                "category": "监管规定",
                "question": "私募基金的风险等级如何划分？",
                "answer": "私募基金按风险程度分为五个等级：\n1. R1（谨慎型）- 风险最低，主要投资债券和现金\n2. R2（稳健型）- 风险较低，混合投资\n3. R3（平衡型）- 风险中等，股债混合\n4. R4（积极型）- 风险较高，主要投资股票\n5. R5（激进型）- 风险最高，可投资衍生品等高风险资产",
                "keywords": ["风险等级", "R1", "R2", "R3", "R4", "R5", "谨慎型", "稳健型", "平衡型", "积极型", "激进型", "等级", "分类"]
            },
            {
                "id": "rule007",
                "category": "监管规定",
                "question": "私募基金管理人应当履行什么责任？",
                "answer": "私募基金管理人的主要责任包括：\n1. 忠实义务 - 恪尽职守，维护投资者利益\n2. 勤勉义务 - 勤勉尽职，制定科学的投资策略\n3. 披露义务 - 及时真实披露基金信息\n4. 风控责任 - 建立有效的风险管理体系\n5. 信息保管 - 保护投资者的个人信息",
                "keywords": ["管理人责任", "忠实义务", "勤勉义务", "披露义务", "风控责任", "信息保管", "管理人", "义务", "责任", "承担"]
            },

            # ========== 信息披露 ==========
            {
                "id": "rule008",
                "category": "信息披露",
                "question": "私募基金需要向投资者披露哪些信息？",
                "answer": "私募基金应当定期向投资者披露以下信息：\n1. 基金净值及单位净值\n2. 投资运作情况\n3. 主要财务指标\n4. 基金经理变更等重大事项\n5. 风险提示信息\n定期披露通常每季度进行一次（书面形式）或每月进行一次（电子形式）",
                "keywords": ["信息披露", "基金净值", "单位净值", "投资运作", "财务指标", "基金经理变更", "重大事项", "风险提示", "季度", "每月", "披露", "定期", "什么", "哪些"]
            },
            {
                "id": "rule009",
                "category": "信息披露",
                "question": "私募基金的基金合同必须包含什么内容？",
                "answer": "基金合同是管理人与投资者的权利义务协议，必须包含：\n1. 当事人的权利和义务\n2. 基金资产的管理、运用、处分原则\n3. 基金资产估值方法\n4. 收益分配原则\n5. 基金的存续期及终止事由\n6. 相关费用的计算和支付\n7. 信息披露方式\n8. 风险揭示等内容",
                "keywords": ["基金合同", "合同内容", "合同要写明", "协议", "主要内容", "权利义务", "资产管理", "估值方法", "收益分配", "存续期", "终止事由", "费用计算", "信息披露方式", "风险揭示"]
            },
            {
                "id": "rule010",
                "category": "信息披露",
                "question": "私募基金需要向监管部门报告什么信息？",
                "answer": "私募基金管理人需要向中国基金业协会进行以下报告：\n1. 定期报告 - 至少每季度提交一次\n2. 重大事项报告 - 15个工作日内报告基金经理变更、重大违约、重大诉讼等\n3. 临时报告 - 对基金产生重大影响的事项\n4. 年度报告 - 包含基金业绩、投资情况等详细信息",
                "keywords": ["监管部门报告", "向协会报告", "向监管部门报告", "定期报告", "重大事项报告", "临时报告", "年度报告", "基金业绩", "协会", "要报告什么信息", "向监管部门要报告"]
            },
            {
                "id": "rule011",
                "category": "投资范围",
                "question": "私募基金可以投资哪些资产？",
                "answer": "私募基金的投资范围包括：\n1. 上市公司股票和非上市企业股权\n2. 债券、票据等固定收益资产\n3. 商品、金融衍生品等\n4. 不动产及其他资产\n5. 法律法规允许的其他资产\n具体的投资范围由基金合同约定，不同类型基金有不同限制",
                "keywords": ["投资资产", "投资范围", "可以投资", "投资什么", "股票", "股权", "债券", "票据", "固定收益", "商品", "衍生品", "不动产", "基金合同约定"]
            },
            {
                "id": "rule012",
                "category": "投资范围",
                "question": "私募基金投资集中度有什么限制？",
                "answer": "为防范风险，私募基金的投资集中度通常受限：\n1. 对单个企业的投资不得超过基金资产总值的20%\n2. 对同一类资产的投资比例受基金合同约束\n3. 与管理人存在关联关系的投资有严格限制\n4. 不得进行法律禁止的投资活动\n这些限制在基金合同中有具体规定",
                "keywords": ["投资集中度", "集中度限制", "单只股票投资", "单个企业", "资产总值", "20%", "关联关系", "法律禁止", "基金合同"]
            },
            {
                "id": "rule013",
                "category": "费用结构",
                "question": "私募基金的费用通常有哪些？",
                "answer": "私募基金的主要费用包括：\n1. 管理费 - 按基金资产净值的一定比例计提（通常1%-2%）\n2. 业绩报酬 - 按超额收益的一定比例计提（通常20%）\n3. 保管费 - 由资产保管人收取（通常0.1%-0.25%）\n4. 其他费用 - 审计费、律师费、信息披露费等\n所有费用应在基金合同中明确列示",
                "keywords": ["费用", "基金费用", "要收什么费用", "付哪些费用", "管理费", "业绩报酬", "保管费", "审计费", "律师费", "信息披露费", "1%-2%", "20%", "0.1%-0.25%"]
            },
            {
                "id": "rule014",
                "category": "费用结构",
                "question": "什么是管理费？如何计算？",
                "answer": "管理费是私募基金管理人因管理基金而获得的报酬。计算方法如下：\n1. 按基金资产净值的年度百分比计提\n2. 通常范围为基金净值的0.5%-3%\n3. 每月计提，按年支付或每年支付一次\n4. 按实际计提日期与当年天数的比例计算\n例如：基金净值1亿元，年费率2%，则年管理费为200万元",
                "keywords": ["管理费", "计算", "基金资产净值", "年度百分比", "0.5%-3%", "每月计提", "按年支付", "比例计算"]
            },
            {
                "id": "rule015",
                "category": "费用结构",
                "question": "什么是业绩报酬？计提条件是什么？",
                "answer": "业绩报酬是管理人获得的超额收益分成。特点如下：\n1. 仅当基金产生正收益时才计提\n2. 通常按超额收益（高于基准收益率的部分）的20%-30%计提\n3. 有些基金采用高水位线机制，确保投资者不重复支付\n4. 计提方式和条件在基金合同中明确规定\n5. 计提时间通常为年度或基金清算时",
                "keywords": ["业绩报酬", "超额收益", "正收益", "20%-30%", "高水位线", "基准收益率", "年度", "清算时", "超额收益分配", "怎么算", "计提条件"]
            },
            {
                "id": "rule016",
                "category": "退出与清算",
                "question": "投资者如何从私募基金中退出？",
                "answer": "投资者的退出方式包括：\n1. 基金清算 - 在基金终止时获得清算收益\n2. 二级市场转让 - 向其他合格投资者转让份额（如基金合同允许）\n3. 管理人回购 - 向基金管理人申请回购（需基金合同同意）\n4. 权益转让 - 将基金权益转让给其他机构\n5. 正常赎回 - 在开放期内赎回（如基金合同约定）\n具体退出方式由基金合同约定",
                "keywords": ["退出", "提前撤资", "清算", "二级市场转让", "管理人回购", "权益转让", "正常赎回", "开放期", "基金合同约定"]
            },
            {
                "id": "rule017",
                "category": "退出与清算",
                "question": "私募基金清算时应该如何分配资产？",
                "answer": "基金清算时的资产分配顺序为：\n1. 支付基金清算费用\n2. 支付基金债务（包括基金保管费、审计费等）\n3. 支付管理人的管理费（如合同要求）\n4. 向投资者返还基金份额对应的净资产\n清算应在基金终止后的规定时间内完成，并将清算报告报送协会",
                "keywords": ["清算", "资产分配", "分钱", "清算费用", "基金债务", "保管费", "审计费", "管理费", "净资产", "清算报告", "协会", "结束了怎么分钱"]
            },
            {
                "id": "rule018",
                "category": "退出与清算",
                "question": "什么情况下私募基金会被强制清算？",
                "answer": "以下情况可能导致私募基金被强制清算：\n1. 基金合同约定的终止事由出现\n2. 基金资产净值持续低于合同约定的最低规模\n3. 管理人被撤销牌照或发生严重违规\n4. 投资者大会或法律规定的其他情形\n5. 基金资产发生重大损失\n强制清算需按法律程序进行，保护投资者合法权益",
                "keywords": ["强制清算", "强行清算", "终止事由", "资产净值", "最低规模", "撤销牌照", "严重违规", "投资者大会", "重大损失", "法律程序", "何时清算", "什么情况清算", "什么情况基金要清算"]
            },
            {
                "id": "rule019",
                "category": "风险管理",
                "question": "私募基金的主要风险有哪些？",
                "answer": "私募基金面临的主要风险包括：\n1. 市场风险 - 投资品种价格波动\n2. 流动性风险 - 基金资产难以变现\n3. 信用风险 - 债务人违约风险\n4. 管理风险 - 管理人操作失误或道德风险\n5. 政策风险 - 法律法规变化\n6. 集中度风险 - 投资过于集中\n投资者需充分了解这些风险",
                "keywords": ["主要风险", "风险", "市场风险", "流动性风险", "信用风险", "管理风险", "政策风险", "集中度风险", "要承担什么风险"]
            },
            {
                "id": "rule020",
                "category": "风险管理",
                "question": "私募基金管理人应该如何进行风险管理？",
                "answer": "私募基金管理人的风险管理措施包括：\n1. 建立完善的风险管理制度和流程\n2. 设置独立的风险管理部门\n3. 定期进行风险评估和压力测试\n4. 建立预警机制和应急预案\n5. 对投资组合进行监测和调整\n6. 计提风险准备金\n7. 进行专业人员培训\n8. 接受外部审计和监督",
                "keywords": ["风险管理", "风险防控", "防控措施", "风险管理制度", "风险管理部门", "风险评估", "压力测试", "预警机制", "应急预案", "投资组合监测", "风险准备金", "人员培训", "外部审计", "怎样进行风险管理"]
            },
            {
                "id": "rule021",
                "category": "合规要求",
                "question": "私募基金管理人需要符合什么合规要求？",
                "answer": "私募基金管理人的合规要求包括：\n1. 建立合规风控部门，配备合规人员\n2. 制定合规管理制度和业务规则\n3. 防范利益冲突，进行关联交易管理\n4. 确保投资者风险评估适当\n5. 防范洗钱和恐怖融资\n6. 保护客户隐私和信息安全\n7. 接受协会自律管理和监管\n8. 定期更新公司治理和内部管理",
                "keywords": ["合规要求", "合规义务", "法律要求", "合规", "风控部门", "合规人员", "管理制度", "利益冲突", "关联交易", "风险评估", "洗钱", "恐怖融资", "客户隐私", "信息安全", "自律管理", "公司治理", "要符合哪些法律要求", "有什么合规义务"]
            },
            {
                "id": "rule022",
                "category": "合规要求",
                "question": "私募基金不能做的事情有什么？",
                "answer": "私募基金禁止的行为包括：\n1. 向不合格投资者募集\n2. 挪用基金资产\n3. 承诺保本保收益\n4. 虚假宣传、误导性宣传\n5. 不公平对待投资者\n6. 从事内幕交易、操纵市场、违法关联交易等。上述行为均属于违法违规。",
                "keywords": ["禁止行为", "不能做的事情", "违法违规", "禁止", "内幕交易", "操纵市场", "违法关联交易", "挪用基金资产", "承诺保本", "虚假宣传", "误导性宣传", "不公平对待", "不合格投资者", "什么是违法违规"]
            }
        ]

    def _initialize_keyword_weights(self) -> Dict[str, int]:
        """初始化关键词权重字典（60+个关键词）"""
        return {
            # 高权重关键词（核心概念）
            "合格投资者": 100,
            "100万元": 95,
            "1000万元": 90,
            "募集规模": 85,
            "管理人资质": 80,
            "风险准备金": 75,
            "10%": 70,
            "风险等级": 65,
            "R1": 60, "R2": 60, "R3": 60, "R4": 60, "R5": 60,
            "忠实义务": 55,
            "勤勉义务": 55,
            "信息披露": 50,
            "基金合同": 45,
            "监管部门报告": 40,
            "投资资产": 35,
            "投资集中度": 30,
            "20%": 25,
            "管理费": 20,
            "业绩报酬": 15,
            "退出": 10,
            "清算": 5,
            "强制清算": 0,

            # 中等权重关键词（相关概念）
            "净资产": 80,
            "金融资产": 75,
            "年均收入": 70,
            "登记": 65,
            "协会": 60,
            "从业资格": 55,
            "财务报告": 50,
            "风险管理体系": 45,
            "牌照": 40,
            "备案": 35,
            "工作日": 30,
            "违法违规": 25,
            "操作错误": 20,
            "谨慎型": 15,
            "稳健型": 10,
            "平衡型": 5,
            "积极型": 0,
            "激进型": -5,
            "披露义务": -10,
            "风控责任": -15,
            "信息保管": -20,
            "基金净值": -25,
            "单位净值": -30,
            "投资运作": -35,
            "财务指标": -40,
            "基金经理变更": -45,
            "重大事项": -50,
            "风险提示": -55,
            "权利义务": -60,
            "资产管理": -65,
            "估值方法": -70,
            "收益分配": -75,
            "存续期": -80,
            "终止事由": -85,
            "费用计算": -90,
            "披露方式": -95,
            "风险揭示": -100,

            # 低权重关键词（辅助概念）
            "定期报告": 50,
            "重大事项报告": 45,
            "临时报告": 40,
            "年度报告": 35,
            "股票": 30,
            "股权": 25,
            "债券": 20,
            "票据": 15,
            "固定收益": 10,
            "商品": 5,
            "衍生品": 0,
            "不动产": -5,
            "单个企业": -10,
            "资产总值": -15,
            "关联关系": -20,
            "法律禁止": -25,
            "保管费": -30,
            "审计费": -35,
            "律师费": -40,
            "披露费": -45,
            "年度百分比": -50,
            "每月计提": -55,
            "按年支付": -60,
            "比例计算": -65,
            "超额收益": -70,
            "正收益": -75,
            "高水位线": -80,
            "基准收益率": -85,
            "二级市场转让": -90,
            "管理人回购": -95,
            "权益转让": -100,
            "正常赎回": -105,
            "开放期": -110,
            "清算费用": -115,
            "基金债务": -120,
            "清算报告": -125,
            "市场风险": -130,
            "流动性风险": -135,
            "信用风险": -140,
            "政策风险": -145,
            "集中度风险": -150,
            "风险管理制度": -155,
            "风险评估": -160,
            "压力测试": -165,
            "预警机制": -170,
            "应急预案": -175,
            "投资组合监测": -180,
            "人员培训": -185,
            "外部审计": -190,
            "合规风控部门": -195,
            "合规人员": -200,
            "关联交易": -205,
            "投资者风险评估": -210,
            "洗钱": -215,
            "恐怖融资": -220,
            "客户隐私": -225,
            "自律管理": -230,
            "公司治理": -235,
            "不合格投资者": -240,
            "挪用基金资产": -245,
            "虚假宣传": -250,
            "误导性宣传": -255,
            "不公平对待": -260,
            "内幕交易": -265,
            "操纵市场": -270,
            "融资融券": -275,
            "杠杆交易": -280,
            "违法关联交易": -285,
            "信息披露不真实": -290
        }

    def _initialize_special_handlers(self) -> Dict[str, callable]:
        """初始化18个特殊处理器"""
        return {
            "qualified_investor_handler": self._handle_qualified_investor,
            "minimum_capital_handler": self._handle_minimum_capital,
            "manager_qualification_handler": self._handle_manager_qualification,
            "raising_period_handler": self._handle_raising_period,
            "risk_reserve_handler": self._handle_risk_reserve,
            "risk_rating_handler": self._handle_risk_rating,
            "manager_responsibility_handler": self._handle_manager_responsibility,
            "disclosure_handler": self._handle_disclosure,
            "contract_content_handler": self._handle_contract_content,
            "regulatory_reporting_handler": self._handle_regulatory_reporting,
            "investment_assets_handler": self._handle_investment_assets,
            "concentration_limit_handler": self._handle_concentration_limit,
            "fee_structure_handler": self._handle_fee_structure,
            "management_fee_handler": self._handle_management_fee,
            "performance_fee_handler": self._handle_performance_fee,
            "exit_mechanism_handler": self._handle_exit_mechanism,
            "liquidation_distribution_handler": self._handle_liquidation_distribution,
            "forced_liquidation_handler": self._handle_forced_liquidation,
            "main_risks_handler": self._handle_main_risks,
            "risk_management_handler": self._handle_risk_management,
            "compliance_handler": self._handle_compliance,
            "prohibited_actions_handler": self._handle_prohibited_actions
        }

    def process_query(self, query: str) -> str:
        """处理用户查询的主方法，所有返回都加【分类】前缀"""
        query = query.strip()
        if not query:
            return "请输入有效的问题。"

        # 步骤1: 关键词匹配和权重计算
        keyword_scores = self._calculate_keyword_scores(query)

        # 步骤2: 语义相似度匹配
        semantic_scores = self._calculate_semantic_scores(query)

        # 步骤3: 综合评分和排序
        combined_scores = self._combine_scores(keyword_scores, semantic_scores)

        # 步骤4: 选择最佳匹配规则
        best_rule_id, best_score = self._select_best_match(combined_scores)

        # 步骤5: 特殊处理器处理
        if best_score > 0:
            rule = self._get_rule_by_id(best_rule_id)
            if rule:
                handler_name = f"{best_rule_id.replace('rule', '').lower()}_handler"
                if handler_name in self.special_handlers:
                    result = self.special_handlers[handler_name](query, rule)
                else:
                    result = self._generate_standard_response(query, rule)
                # 若未加【分类】，则补加
                if f"【{rule['category']}】" not in result:
                    result = f"【{rule['category']}】" + result
                return result

        # 步骤6: LLM增强回答（当匹配度不高时）
        # 兜底也加上分类
        rule = self._get_rule_by_id(best_rule_id) if best_rule_id else None
        category = rule['category'] if rule and 'category' in rule else '其他'
        result = self._generate_llm_enhanced_response(query)
        if f"【{category}】" not in result:
            result = f"【{category}】" + result
        return result

    def _generate_standard_response(self, query: str, rule: Dict[str, Any]) -> str:
        """生成标准回答，带分类"""
        return f"【{rule['category']}】{rule['question']}\n\n{rule['answer']}"

    def _generate_llm_enhanced_response(self, query: str) -> str:
        """使用LLM生成增强回答"""
        # 找到最相关的规则
        keyword_scores = self._calculate_keyword_scores(query)
        semantic_scores = self._calculate_semantic_scores(query)
        combined_scores = self._combine_scores(keyword_scores, semantic_scores)
        best_rule_id, _ = self._select_best_match(combined_scores)

        if best_rule_id:
            rule = self._get_rule_by_id(best_rule_id)
            if rule:
                prompt = ChatPromptTemplate.from_template("""
                你是私募基金问答助手，请回答用户的问题。
                相关规则：{question}
                规则内容：{answer}

                用户问题：{query}

                请基于规则内容提供准确回答：
                """)

                chain = prompt | self.llm | StrOutputParser()
                llm_answer = chain.invoke({
                    "question": rule['question'],
                    "answer": rule['answer'],
                    "query": query
                })

                return f"{rule['question']}\n\n{llm_answer}"

        # 如果没有找到相关规则，使用通用回答
        prompt = ChatPromptTemplate.from_template("""
        你是私募基金问答助手，请回答用户关于私募基金的问题。

        用户问题：{query}

        请提供专业回答：
        """)

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})

    def _calculate_keyword_scores(self, query: str) -> Dict[str, float]:
        """计算关键词匹配分数"""
        query_lower = query.lower()
        scores = {}

        for rule in self.rules_db:
            rule_id = rule['id']
            matched_count = 0

            for kw in rule.get('keywords', []):
                if kw.lower() in query_lower:
                    matched_count += 1

            scores[rule_id] = matched_count

        return scores

    def _calculate_semantic_scores(self, query: str) -> Dict[str, float]:
        """计算语义相似度分数"""
        scores = {}

        for rule in self.rules_db:
            rule_id = rule['id']
            question = rule['question']

            # 使用序列匹配器计算相似度
            similarity = SequenceMatcher(None, query.lower(), question.lower()).ratio()

            scores[rule_id] = similarity

        return scores

    def _combine_scores(self, keyword_scores: Dict[str, float], semantic_scores: Dict[str, float]) -> Dict[str, float]:
        """综合关键词和语义分数，优先关键词数量，再用语义分数辅助"""
        max_kw = max(keyword_scores.values())
        candidates = {rid: score for rid, score in keyword_scores.items() if score == max_kw and score > 0}
        if len(candidates) == 1:
            return candidates
        # 多个并列时用语义分数区分
        if candidates:
            best_sem = max(candidates, key=lambda rid: semantic_scores.get(rid, 0))
            return {best_sem: keyword_scores[best_sem]}
        # 没有关键词命中时，退化为语义分数最高
        best_sem = max(semantic_scores, key=lambda rid: semantic_scores[rid])
        return {best_sem: 0}

    def _select_best_match(self, scores: Dict[str, float]) -> Tuple[str, float]:
        """选择最佳匹配规则"""
        if not scores:
            return "", 0.0

        # 按分数降序，ID升序排序，选择第一个
        sorted_candidates = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        best_rule_id, best_score = sorted_candidates[0]
        return best_rule_id, best_score

    def _get_rule_by_id(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取规则"""
        for rule in self.rules_db:
            if rule['id'] == rule_id:
                return rule
        return None

    def _generate_standard_response(self, query: str, rule: Dict[str, Any]) -> str:
        """生成标准回答"""
        return f"{rule['question']}\n\n{rule['answer']}"

    def _generate_llm_enhanced_response(self, query: str) -> str:
        """使用LLM生成增强回答"""
        # 查找最相关的规则作为上下文
        best_rule = None
        best_similarity = 0

        for rule in self.rules_db:
            similarity = SequenceMatcher(None, query.lower(), rule['question'].lower()).ratio()
            if similarity > best_similarity:
                best_similarity = similarity
                best_rule = rule

        if best_rule:
            context = best_rule['answer']
        else:
            context = "根据私募基金相关法规和行业实践"

        prompt = f"""你是一个专业的私募基金问答助手。请基于以下信息回答用户的问题。

信息：{context}

用户问题：{query}

请提供准确、专业、简洁的回答。如果信息不足，请明确说明。"""

        try:
            response = self.llm(prompt)
            return response.strip()
        except Exception as e:
            return f"抱歉，处理您的问题时出现错误：{str(e)}"

    # ========== 特殊处理器实现 ==========

    def _handle_qualified_investor(self, query: str, rule: Dict[str, Any]) -> str:
        """处理合格投资者相关查询"""
        if "最低" in query or "多少" in query:
            return f"【{rule['category']}】{rule['question']}\n\n合格投资者的最低投资金额为100万元。"
        elif "条件" in query:
            return f"【{rule['category']}】{rule['question']}\n\n{rule['answer']}"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_minimum_capital(self, query: str, rule: Dict[str, Any]) -> str:
        """处理最低募集规模查询"""
        if "多少" in query or "金额" in query:
            return f"【{rule['category']}】{rule['question']}\n\n私募证券投资基金的最低募集规模为1000万元人民币。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_manager_qualification(self, query: str, rule: Dict[str, Any]) -> str:
        """处理管理人资质查询"""
        if "怎么" in query or "怎样" in query:
            return f"【{rule['category']}】{rule['question']}\n\n成为私募基金管理人需要在中国证券投资基金业协会登记，满足公司运营两年、高管从业资格、财务报告和风险管理体系等条件。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_raising_period(self, query: str, rule: Dict[str, Any]) -> str:
        """处理募集期查询"""
        if "多长" in query or "时间" in query:
            return f"【{rule['category']}】{rule['question']}\n\n私募基金的募集期通常为6个月。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_risk_reserve(self, query: str, rule: Dict[str, Any]) -> str:
        """处理风险准备金查询"""
        if "多少" in query or "比例" in query:
            return f"【{rule['category']}】{rule['question']}\n\n私募基金管理人需要按照管理费收入的10%计提风险准备金。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_risk_rating(self, query: str, rule: Dict[str, Any]) -> str:
        """处理风险等级查询"""
        if "R1" in query or "R2" in query or "R3" in query or "R4" in query or "R5" in query:
            return f"【{rule['category']}】{rule['question']}\n\n{rule['answer']}"
        else:
            return f"【{rule['category']}】{rule['question']}\n\n私募基金按风险程度分为R1（谨慎型）到R5（激进型）五个等级。"

    def _handle_manager_responsibility(self, query: str, rule: Dict[str, Any]) -> str:
        """处理管理人责任查询"""
        if "义务" in query:
            return f"【{rule['category']}】{rule['question']}\n\n私募基金管理人需要履行忠实义务、勤勉义务、披露义务、风控责任和信息保管责任。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_disclosure(self, query: str, rule: Dict[str, Any]) -> str:
        """处理信息披露查询"""
        if "什么" in query or "哪些" in query:
            return f"【{rule['category']}】{rule['question']}\n\n私募基金需要披露基金净值、投资运作情况、财务指标、重大事项和风险提示信息。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_contract_content(self, query: str, rule: Dict[str, Any]) -> str:
        """处理基金合同内容查询"""
        if "必须" in query or "包含" in query:
            return f"【{rule['category']}】{rule['question']}\n\n基金合同必须包含当事人的权利义务、资产管理原则、估值方法、收益分配、存续期、费用计算、信息披露方式和风险揭示等内容。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_regulatory_reporting(self, query: str, rule: Dict[str, Any]) -> str:
        """处理监管报告查询"""
        if "什么" in query or "哪些" in query:
            return f"【{rule['category']}】{rule['question']}\n\n私募基金需要向协会提交定期报告、重大事项报告、临时报告和年度报告。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_investment_assets(self, query: str, rule: Dict[str, Any]) -> str:
        """处理投资资产查询"""
        if "哪些" in query or "什么" in query:
            return f"【{rule['category']}】{rule['question']}\n\n私募基金可以投资股票、股权、债券、票据、商品、衍生品、不动产等资产，具体范围由基金合同约定。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_concentration_limit(self, query: str, rule: Dict[str, Any]) -> str:
        """处理集中度限制查询"""
        if "多少" in query or "比例" in query:
            return f"【{rule['category']}】{rule['question']}\n\n私募基金对单个企业的投资不得超过基金资产总值的20%。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_fee_structure(self, query: str, rule: Dict[str, Any]) -> str:
        """处理费用结构查询"""
        if "哪些" in query or "有什么" in query:
            return f"【{rule['category']}】{rule['question']}\n\n私募基金的主要费用包括管理费（1%-2%）、业绩报酬（20%）、保管费（0.1%-0.25%）和其他费用。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_management_fee(self, query: str, rule: Dict[str, Any]) -> str:
        """处理管理费查询"""
        if "怎么" in query or "如何" in query:
            return f"【{rule['category']}】{rule['question']}\n\n管理费按基金资产净值的年度百分比计提，通常为0.5%-3%，每月计提，按年支付。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_performance_fee(self, query: str, rule: Dict[str, Any]) -> str:
        """处理业绩报酬查询"""
        if "条件" in query or "怎么" in query:
            return f"【{rule['category']}】{rule['question']}\n\n业绩报酬仅在基金产生正收益时计提，通常按超额收益的20%-30%计提，有些基金采用高水位线机制。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_exit_mechanism(self, query: str, rule: Dict[str, Any]) -> str:
        """处理退出机制查询"""
        if "怎么" in query or "如何" in query:
            return f"【{rule['category']}】{rule['question']}\n\n投资者可以通过基金清算、二级市场转让、管理人回购、权益转让或正常赎回等方式退出私募基金。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_liquidation_distribution(self, query: str, rule: Dict[str, Any]) -> str:
        """处理清算分配查询"""
        if "顺序" in query or "如何" in query:
            return f"【{rule['category']}】{rule['question']}\n\n清算时资产分配顺序为：清算费用→基金债务→管理费（如合同要求）→向投资者返还净资产。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_forced_liquidation(self, query: str, rule: Dict[str, Any]) -> str:
        """处理强制清算查询"""
        if "什么" in query or "哪些" in query:
            return f"【{rule['category']}】{rule['question']}\n\n基金合同终止事由、资产净值低于最低规模、管理人被撤销牌照、严重违规或重大损失等情况可能导致强制清算。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_main_risks(self, query: str, rule: Dict[str, Any]) -> str:
        """处理主要风险查询"""
        if "哪些" in query or "有什么" in query:
            return f"【{rule['category']}】{rule['question']}\n\n私募基金的主要风险包括市场风险、流动性风险、信用风险、管理风险、政策风险和集中度风险。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_risk_management(self, query: str, rule: Dict[str, Any]) -> str:
        """处理风险管理查询"""
        if "怎么" in query or "如何" in query:
            return f"【{rule['category']}】{rule['question']}\n\n管理人应建立风险管理制度、设置风险部门、进行风险评估、建立预警机制、监测投资组合和计提风险准备金等。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_compliance(self, query: str, rule: Dict[str, Any]) -> str:
        """处理合规要求查询"""
        if "什么" in query or "哪些" in query:
            return f"【{rule['category']}】{rule['question']}\n\n管理人需要建立合规部门、制定管理制度、防范利益冲突、确保风险评估适当、防范洗钱和保护信息安全等。"
        else:
            return self._generate_standard_response(query, rule)

    def _handle_prohibited_actions(self, query: str, rule: Dict[str, Any]) -> str:
        """处理禁止行为查询"""
        if "什么" in query or "哪些" in query:
            return f"【{rule['category']}】{rule['question']}\n\n禁止行为包括向不合格投资者募集、挪用基金资产、承诺保本保收益、虚假宣传、内幕交易、操纵市场和违法关联交易等。"
        else:
            return self._generate_standard_response(query, rule)

    def _generate_llm_enhanced_response(self, query: str) -> str:
        """使用LLM生成增强回答"""
        # 查找最相关的规则作为上下文
        best_rule = None
        best_similarity = 0

        for rule in self.rules_db:
            similarity = SequenceMatcher(None, query.lower(), rule['question'].lower()).ratio()
            if similarity > best_similarity:
                best_similarity = similarity
                best_rule = rule

        if best_rule:
            return f"【{best_rule['category']}】{best_rule['question']}\n\n{best_rule['answer']}"
        else:
            return "抱歉，我无法找到相关信息。"

if __name__ == "__main__":
    assistant = FundQAAssistant()
    print("私募基金问答助手已启动，输入 exit 退出。\n")
    while True:
        try:
            query = input("请输入您的私募基金问题：")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if query.strip().lower() == "exit":
            print("已退出。")
            break
        if not query.strip():
            continue
        answer = assistant.process_query(query)
        print(f"\n{answer}\n")