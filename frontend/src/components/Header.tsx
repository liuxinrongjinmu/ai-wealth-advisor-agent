import { Cpu } from 'lucide-react'
import { useChatStore, type TabType } from '../store'

const tabs: { key: TabType; label: string; icon: string }[] = [
  { key: 'fund-qa', label: '基金问答', icon: '📋' },
  { key: 'research', label: '投研分析', icon: '🔬' },
  { key: 'wealth-advisor', label: '财富顾问', icon: '💎' },
]

export default function Header() {
  const { activeTab, setActiveTab } = useChatStore()

  return (
    <header className="glass-heavy sticky top-0 z-50 border-b border-white/30">
      <div className="mx-auto flex h-16 max-w-4xl items-center justify-between px-5">
        {/* Logo区域 */}
        <div className="flex items-center gap-3">
          <div className="glass-light flex h-9 w-9 items-center justify-center rounded-lg">
            <Cpu className="h-5 w-5 text-accent" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-wide text-text-primary">
              智能投顾<span className="text-accent">AI</span>助手
            </h1>
            <div className="tech-line mt-0.5 w-full" />
          </div>
        </div>

        {/* 导航标签 */}
        <nav className="glass-light flex gap-1 rounded-xl p-1">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-sm font-medium transition-all duration-200 ${
                activeTab === tab.key
                  ? 'bg-white/80 text-accent shadow-sm ring-1 ring-accent/15 backdrop-blur-sm'
                  : 'text-text-secondary hover:text-text-primary hover:bg-white/40'
              }`}
            >
              <span className="text-xs">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>
      </div>
    </header>
  )
}
