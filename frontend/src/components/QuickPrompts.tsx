import { useChatStore, type TabType } from '../store'

const prompts: Record<TabType, string[]> = {
  'fund-qa': [
    '合格投资者标准是什么？',
    '私募基金有哪些费用？',
    '风险等级如何划分？',
    '禁止行为有哪些？',
  ],
  research: [
    '新能源汽车行业投资机会',
    '人工智能行业分析',
    '消费行业投资前景',
  ],
  'wealth-advisor': [
    '今天上证指数怎么样？',
    '如何调整投资组合？',
    '什么是ETF？',
    '退休规划建议',
  ],
}

interface Props {
  onSelect: (prompt: string) => void
}

export default function QuickPrompts({ onSelect }: Props) {
  const activeTab = useChatStore((s) => s.activeTab)
  const items = prompts[activeTab]

  return (
    <div className="flex flex-wrap gap-2">
      {items.map((prompt, idx) => (
        <button
          key={prompt}
          onClick={() => onSelect(prompt)}
          className="glass group flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs text-text-secondary transition-all duration-200 hover:text-accent hover:shadow-md hover:shadow-accent/8"
        >
          <span className="text-accent/40 group-hover:text-accent transition-colors font-mono">
            {String(idx + 1).padStart(2, '0')}
          </span>
          <span className="tech-line-vertical h-3" />
          {prompt}
        </button>
      ))}
    </div>
  )
}
