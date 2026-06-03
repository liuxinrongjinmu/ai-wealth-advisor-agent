import { useChatStore, type TabKey } from '../store/chatStore'
import { Zap } from 'lucide-react'

interface PromptOption {
  label: string
  query: string
}

const prompts: Record<TabKey, PromptOption[]> = {
  'fund-qa': [
    { label: '合格投资者标准', query: '私募基金的合格投资者标准是什么？' },
    { label: '投资范围', query: '私募基金可以投资哪些资产？' },
    { label: '费用结构', query: '私募基金有哪些费用？' },
    { label: '禁止行为', query: '私募基金有什么禁止的行为？' },
  ],
  research: [
    { label: '新能源投资机会', query: '新能源汽车行业投资机会分析' },
    { label: 'AI产业趋势', query: '人工智能产业未来发展趋势' },
    { label: '半导体行业', query: '半导体行业投资价值评估' },
  ],
  'wealth-advisor': [
    { label: '投资组合优化', query: '如何优化我的投资组合？' },
    { label: '风险管理', query: '当前市场环境下如何控制风险？' },
    { label: '今日行情', query: '今天上证指数表现如何？' },
    { label: '退休规划', query: '如何为退休做长期投资规划？' },
  ],
}

interface Props {
  onSelect: (query: string) => void
}

export default function QuickPrompts({ onSelect }: Props) {
  const activeTab = useChatStore((s) => s.activeTab)
  const currentPrompts = prompts[activeTab]

  return (
    <div className="flex flex-wrap gap-2 px-1">
      <Zap size={14} className="mt-0.5 shrink-0 text-amber-400" />
      {currentPrompts.map((p) => (
        <button
          key={p.label}
          onClick={() => onSelect(p.query)}
          className="rounded-full border border-slate-600 bg-slate-800/50 px-3 py-1 text-xs text-slate-300
                     transition-all duration-200 hover:border-amber-500/50 hover:bg-amber-500/10 hover:text-amber-300
                     active:scale-95"
        >
          {p.label}
        </button>
      ))}
    </div>
  )
}