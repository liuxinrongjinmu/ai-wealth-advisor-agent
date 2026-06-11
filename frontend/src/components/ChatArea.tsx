import { useEffect, useRef, useState } from 'react'
import { Cpu } from 'lucide-react'
import { useChatStore } from '../store'
import MessageBubble from './MessageBubble'

const loadingMessages = [
  '正在思考中...',
  '正在查询相关知识...',
  '正在生成回答...',
  '请稍候，AI正在处理...',
]

interface Props {
  researchIndustry: string
  researchHorizon: string
  onIndustryChange: (v: string) => void
  onHorizonChange: (v: string) => void
}

export default function ChatArea({ researchIndustry, researchHorizon, onIndustryChange, onHorizonChange }: Props) {
  const { messages, isLoading, activeTab } = useChatStore()
  const bottomRef = useRef<HTMLDivElement>(null)
  const [loadingText, setLoadingText] = useState(loadingMessages[0])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  useEffect(() => {
    if (!isLoading) {
      setLoadingText(loadingMessages[0])
      return
    }
    let idx = 0
    const timer = setInterval(() => {
      idx = (idx + 1) % loadingMessages.length
      setLoadingText(loadingMessages[idx])
    }, 4000)
    return () => clearInterval(timer)
  }, [isLoading])

  const tabLabels = {
    'fund-qa': '私募基金问答',
    research: '投研分析',
    'wealth-advisor': '财富顾问',
  }

  const tabDescriptions = {
    'fund-qa': '基于22条私募基金核心规则的智能匹配问答',
    research: '五阶段深度分析：感知→建模→推理→决策→报告',
    'wealth-advisor': '智能路由：简单查询快速响应，复杂查询深度分析',
  }

  const industries = ['综合', '新能源', '人工智能', '消费', '医药', '半导体', '金融科技', '高端制造']
  const horizons = ['短期', '中期', '长期']

  return (
    <div className="chat-scroll flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl px-5 py-6">
        {messages.length === 0 && !isLoading && (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            {/* 磨砂玻璃图标容器 */}
            <div className="glass-card mb-6 flex h-20 w-20 items-center justify-center rounded-2xl">
              <Cpu className="h-10 w-10 text-accent" />
            </div>
            <h2 className="mb-2 text-2xl font-bold text-text-primary tracking-wide">
              {tabLabels[activeTab]}
            </h2>
            <p className="mb-3 max-w-md text-sm text-text-secondary leading-relaxed">
              {tabDescriptions[activeTab]}
            </p>

            {/* 投研分析模式：显示行业和时间范围选择器 */}
            {activeTab === 'research' && (
              <div className="mb-4 flex items-center gap-3">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-text-muted">行业：</span>
                  <select
                    value={researchIndustry}
                    onChange={(e) => onIndustryChange(e.target.value)}
                    className="glass rounded-lg border border-border px-2.5 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent/50 cursor-pointer"
                  >
                    {industries.map((ind) => (
                      <option key={ind} value={ind}>{ind}</option>
                    ))}
                  </select>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-text-muted">期限：</span>
                  <select
                    value={researchHorizon}
                    onChange={(e) => onHorizonChange(e.target.value)}
                    className="glass rounded-lg border border-border px-2.5 py-1.5 text-xs text-text-primary focus:outline-none focus:border-accent/50 cursor-pointer"
                  >
                    {horizons.map((h) => (
                      <option key={h} value={h}>{h}</option>
                    ))}
                  </select>
                </div>
              </div>
            )}

            <div className="tech-line w-32" />
          </div>
        )}
        <div className="space-y-4">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          {isLoading && (
            <div className="message-animate flex justify-start">
              <div className="glass-card flex items-center gap-2.5 rounded-2xl px-4 py-3">
                <span className="dot-pulse h-2 w-2 rounded-full bg-accent" />
                <span className="dot-pulse h-2 w-2 rounded-full bg-accent" />
                <span className="dot-pulse h-2 w-2 rounded-full bg-accent" />
                <span className="ml-1 text-xs text-accent">{loadingText}</span>
              </div>
            </div>
          )}
        </div>
        <div ref={bottomRef} />
      </div>
    </div>
  )
}