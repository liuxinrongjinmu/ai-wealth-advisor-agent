import { useEffect, useRef, useState } from 'react'
import { Cpu, CheckCircle2, Loader2, Circle, ClipboardList, Search, Lightbulb, Users, ArrowUpRight } from 'lucide-react'
import { useChatStore } from '../store'
import MessageBubble from './MessageBubble'
import RiskQuestionnaire from './RiskQuestionnaire'
import { MessageSkeleton } from './Skeleton'

const loadingMessages = [
  '正在思考中...',
  '正在查询相关知识...',
  '正在生成回答...',
  '请稍候，AI正在处理...',
]

/** 各Tab的快捷提问示例 */
const examplePrompts: Record<string, string[]> = {
  'fund-qa': [
    '合格投资者标准是什么？',
    '私募基金有哪些费用？',
    '风险等级如何划分？',
    '信息披露有哪些要求？',
  ],
  research: [
    '新能源汽车行业投资机会',
    '人工智能行业分析',
    '消费行业投资前景',
  ],
  'wealth-advisor': [
    '如何调整投资组合？',
    '退休规划建议',
    '资产配置策略',
  ],
}

interface Props {
  researchIndustry: string
  researchHorizon: string
  onIndustryChange: (v: string) => void
  onHorizonChange: (v: string) => void
  onRegenerate: () => void
  /** 点击快捷提问回调 */
  onSendPrompt?: (prompt: string) => void
}

const SCENE_GUIDES = {
  'fund-qa': {
    title: '私募基金合规问答',
    desc: '基于22条核心规则的智能匹配问答，快速获取法规合规指引',
    icon: Search,
    color: 'text-blue-500',
    bg: 'bg-blue-50',
    suits: ['合规投资人条件查询', '基金设立与募集要求', '信息披露规定', '投资范围与限制'],
    notSuits: ['投资策略分析', '基金产品推荐', '实时行情查询'],
  },
  research: {
    title: '智能投研分析',
    desc: '五阶段深度分析：感知→建模→推理→决策→报告，生成专业投研报告',
    icon: Lightbulb,
    color: 'text-accent',
    bg: 'bg-accent/10',
    suits: ['行业投资机会分析', '主题研究报告', '投资逻辑推演', '多行业对比分析'],
    notSuits: ['快速法规查询', '实时基金净值查询', '客户资产配置建议'],
  },
  'wealth-advisor': {
    title: '财富管理顾问',
    desc: '智能路由：简单查询快速响应，复杂查询深度分析，支持个性化客户画像',
    icon: Users,
    color: 'text-purple-500',
    bg: 'bg-purple-50',
    suits: ['资产配置建议', '风险评估问卷', '客户投资咨询', '持仓分析讨论'],
    notSuits: ['法规条文逐条解释', '市场行情预测', '具体基金产品推荐'],
  },
}

export default function ChatArea({ researchIndustry, researchHorizon, onIndustryChange, onHorizonChange, onRegenerate, onSendPrompt }: Props) {
  const { messages, isLoading, isStreaming, researchStages, activeTab, customerId } = useChatStore()
  const bottomRef = useRef<HTMLDivElement>(null)
  const [loadingText, setLoadingText] = useState(loadingMessages[0])
  const [showQuestionnaire, setShowQuestionnaire] = useState(false)
  /** 客户画像摘要 */
  const [customerProfile, setCustomerProfile] = useState<Record<string, any> | null>(null)

  // 加载客户画像
  useEffect(() => {
    if (activeTab === 'wealth-advisor') {
      fetch(`/api/v1/customers/${customerId}`)
        .then((r) => r.ok ? r.json() : null)
        .then(setCustomerProfile)
        .catch(() => setCustomerProfile(null))
    }
  }, [activeTab, customerId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading, researchStages])

  useEffect(() => {
    if (!isLoading && !isStreaming) {
      setLoadingText(loadingMessages[0])
      return
    }
    let idx = 0
    const timer = setInterval(() => {
      idx = (idx + 1) % loadingMessages.length
      setLoadingText(loadingMessages[idx])
    }, 4000)
    return () => clearInterval(timer)
  }, [isLoading, isStreaming])

  const guide = SCENE_GUIDES[activeTab]
  const GuideIcon = guide.icon
  const industries = ['综合', '新能源', '人工智能', '消费', '医药', '半导体', '金融科技', '高端制造']
  const horizons = ['短期', '中期', '长期']

  const lastMsgIndex = messages.length - 1
  // 最后一条AI消息才显示重新生成按钮
  const lastAssistantIdx = [...messages].reverse().findIndex((m) => m.role === 'assistant')

  return (
    <div className="chat-scroll flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl px-5 py-6">
        {messages.length === 0 && !isLoading && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            {/* 场景引导卡片 */}
            <div className="glass-card mb-6 max-w-md w-full rounded-2xl p-6">
              <div className={`mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl ${guide.bg}`}>
                <GuideIcon className={`h-7 w-7 ${guide.color}`} />
              </div>
              <h2 className="mb-1.5 text-xl font-bold text-text-primary tracking-wide">
                {guide.title}
              </h2>
              <p className="mb-5 text-sm text-text-secondary leading-relaxed">
                {guide.desc}
              </p>

              {/* 适用 / 不适用 */}
              <div className="grid grid-cols-2 gap-3 text-left">
                <div className="rounded-xl bg-green-50/60 p-3">
                  <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-green-600">适用场景</p>
                  <ul className="space-y-1">
                    {guide.suits.map((s) => (
                      <li key={s} className="text-xs text-green-700 flex items-center gap-1">
                        <span className="h-1 w-1 rounded-full bg-green-400 shrink-0" />
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="rounded-xl bg-red-50/60 p-3">
                  <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-red-500">不适用场景</p>
                  <ul className="space-y-1">
                    {guide.notSuits.map((s) => (
                      <li key={s} className="text-xs text-red-600 flex items-center gap-1">
                        <span className="h-1 w-1 rounded-full bg-red-300 shrink-0" />
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>

            {activeTab === 'research' && (
              <div className="mb-4 flex items-center gap-3">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-text-muted">行业：</span>
                  <select value={researchIndustry} onChange={(e) => onIndustryChange(e.target.value)}
                    className="glass rounded-lg border border-border px-2.5 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent/50 cursor-pointer">
                    {industries.map((ind) => (<option key={ind} value={ind}>{ind}</option>))}
                  </select>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-text-muted">期限：</span>
                  <select value={researchHorizon} onChange={(e) => onHorizonChange(e.target.value)}
                    className="glass rounded-lg border border-border px-2.5 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent/50 cursor-pointer">
                    {horizons.map((h) => (<option key={h} value={h}>{h}</option>))}
                  </select>
                </div>
              </div>
            )}

            {/* 财富顾问：客户画像摘要卡片 */}
            {activeTab === 'wealth-advisor' && customerProfile && (
              <div className="glass-card mb-4 max-w-md w-full rounded-xl p-4 text-left">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-purple-500 mb-2">当前客户画像</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {customerProfile.risk_level && (
                    <div className="flex items-center gap-1.5">
                      <span className={`h-2 w-2 rounded-full ${
                        customerProfile.risk_level === 'R1' || customerProfile.risk_level === 'R2' ? 'bg-green-400' :
                        customerProfile.risk_level === 'R3' ? 'bg-yellow-400' : 'bg-red-400'
                      }`} />
                      <span className="text-text-secondary">风险等级：</span>
                      <span className="text-text-primary font-medium">{customerProfile.risk_level}</span>
                    </div>
                  )}
                  {customerProfile.investment_horizon && (
                    <div className="flex items-center gap-1.5">
                      <span className="text-text-secondary">投资期限：</span>
                      <span className="text-text-primary font-medium">{customerProfile.investment_horizon}</span>
                    </div>
                  )}
                  {customerProfile.name && (
                    <div className="flex items-center gap-1.5">
                      <span className="text-text-secondary">姓名：</span>
                      <span className="text-text-primary font-medium">{customerProfile.name}</span>
                    </div>
                  )}
                  {customerProfile.age && (
                    <div className="flex items-center gap-1.5">
                      <span className="text-text-secondary">年龄：</span>
                      <span className="text-text-primary font-medium">{customerProfile.age}岁</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* 财富顾问：风险评估问卷入口 */}
            {activeTab === 'wealth-advisor' && !showQuestionnaire && (
              <button onClick={() => setShowQuestionnaire(true)}
                className="glass mb-4 flex items-center gap-2 rounded-xl px-4 py-2.5 text-xs text-accent hover:border-accent/40 transition-all duration-200 border border-transparent hover:border-accent/20">
                <ClipboardList className="h-4 w-4" />
                开始风险评估，获取个性化投资建议
              </button>
            )}

            {/* 快捷提问按钮（点击即可发送） */}
            {onSendPrompt && examplePrompts[activeTab] && (
              <div className="mb-6 w-full max-w-md">
                <p className="text-[10px] text-text-muted mb-2 font-medium uppercase tracking-wider">快速开始</p>
                <div className="flex flex-wrap gap-2 justify-center">
                  {examplePrompts[activeTab].map((prompt, idx) => (
                    <button
                      key={prompt}
                      onClick={() => onSendPrompt(prompt)}
                      className="glass group flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs text-text-secondary transition-all duration-200 hover:text-accent hover:shadow-md hover:shadow-accent/10 hover:border-accent/20 border border-transparent"
                    >
                      <span className="text-accent/40 group-hover:text-accent transition-colors font-mono text-[10px]">
                        {(idx + 1).toString().padStart(2, '0')}
                      </span>
                      {prompt}
                      <ArrowUpRight className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="tech-line w-32" />
          </div>
        )}
        <div className="space-y-4">
          {/* 风险评估问卷 */}
          {showQuestionnaire && activeTab === 'wealth-advisor' && (
            <RiskQuestionnaire
              onComplete={() => setShowQuestionnaire(false)}
              onClose={() => setShowQuestionnaire(false)}
            />
          )}
          {messages.map((msg, idx) => (
            <MessageBubble
              key={msg.id}
              message={msg}
              showRegenerate={!isLoading && msg.role === 'assistant' && idx === lastMsgIndex}
              onRegenerate={onRegenerate}
            />
          ))}

          {/* 投研流式阶段进度 */}
          {isStreaming && researchStages.length > 0 && (
            <div className="message-animate flex justify-start">
              <div className="glass-card max-w-[80%] rounded-2xl px-5 py-4">
                <div className="mb-3 flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-accent float-glow" />
                  <span className="text-[10px] font-medium uppercase tracking-widest text-accent">
                    投研进度
                  </span>
                </div>
                <div className="space-y-2">
                  {researchStages.map((stage) => (
                    <div key={stage.key} className="flex items-center gap-2.5">
                      {stage.status === 'done' ? (
                        <CheckCircle2 className="h-4 w-4 text-success shrink-0" />
                      ) : stage.status === 'running' ? (
                        <Loader2 className="h-4 w-4 text-accent animate-spin shrink-0" />
                      ) : (
                        <Circle className="h-4 w-4 text-text-muted shrink-0" />
                      )}
                      <span className={`text-xs ${
                        stage.status === 'done' ? 'text-success' :
                        stage.status === 'running' ? 'text-accent font-medium' :
                        'text-text-muted'
                      }`}>
                        {stage.name} · {stage.message}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* 普通加载动画（非流式模式）- 使用骨架屏 */}
          {isLoading && !isStreaming && <MessageSkeleton />}
        </div>
        <div ref={bottomRef} />
      </div>
    </div>
  )
}