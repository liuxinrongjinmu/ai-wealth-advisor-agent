import { useState, useRef, useEffect } from 'react'
import { Send, Sparkles } from 'lucide-react'
import { useChatStore } from '../store'
import QuickPrompts from './QuickPrompts'

interface Props {
  onSend: (message: string) => void
  researchIndustry: string
  researchHorizon: string
  onIndustryChange: (v: string) => void
  onHorizonChange: (v: string) => void
}

export default function InputArea({ onSend, researchIndustry, researchHorizon, onIndustryChange, onHorizonChange }: Props) {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const { isLoading, messages, activeTab, customerId, setCustomerId } = useChatStore()

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px'
    }
  }, [input])

  const handleSend = () => {
    const trimmed = input.trim()
    if (!trimmed || isLoading) return
    onSend(trimmed)
    setInput('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const industries = ['综合', '新能源', '人工智能', '消费', '医药', '半导体', '金融科技', '高端制造']
  const horizons = ['短期', '中期', '长期']

  return (
    <div className="glass-heavy border-t border-white/30">
      <div className="mx-auto max-w-4xl px-5 py-4">
        {/* 快捷提问 */}
        {messages.length === 0 && <QuickPrompts onSelect={(p) => onSend(p)} />}

        {/* 投研分析模式选项栏 */}
        {activeTab === 'research' && messages.length > 0 && (
          <div className="mb-2 flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-text-muted">行业：</span>
              <select value={researchIndustry} onChange={(e) => onIndustryChange(e.target.value)}
                className="glass rounded-lg border border-border px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent/50 cursor-pointer">
                {industries.map((ind) => (<option key={ind} value={ind}>{ind}</option>))}
              </select>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-text-muted">期限：</span>
              <select value={researchHorizon} onChange={(e) => onHorizonChange(e.target.value)}
                className="glass rounded-lg border border-border px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent/50 cursor-pointer">
                {horizons.map((h) => (<option key={h} value={h}>{h}</option>))}
              </select>
            </div>
          </div>
        )}

        {/* 财富顾问模式客户选择 */}
        {activeTab === 'wealth-advisor' && (
          <div className="mb-2 flex items-center gap-1.5">
            <span className="text-xs text-text-muted">客户：</span>
            <select value={customerId} onChange={(e) => setCustomerId(e.target.value)}
              className="glass rounded-lg border border-border px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent/50 cursor-pointer">
              <option value="customer1">张三（平衡型）</option>
              <option value="customer2">李四（进取型）</option>
            </select>
          </div>
        )}

        <div className="input-glow glass mt-2 flex items-end gap-2 rounded-xl p-2 transition-all duration-200">
          <div className="glass-light flex h-9 w-9 shrink-0 items-center justify-center rounded-lg">
            <Sparkles className="h-4 w-4 text-accent" />
          </div>
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入您的问题..."
            rows={1}
            disabled={isLoading}
            className="flex-1 resize-none bg-transparent px-1 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-accent to-accent-dark text-white transition-all duration-200 hover:shadow-lg hover:shadow-accent/20 disabled:opacity-30 disabled:hover:shadow-none"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
        <p className="mt-2 text-center text-[10px] text-text-muted">
          基于 LangChain + LangGraph + Qwen-Turbo 构建 · 回答仅供参考，不构成投资建议
        </p>
      </div>
    </div>
  )
}