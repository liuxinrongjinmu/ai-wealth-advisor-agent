import { useChatStore, type TabKey } from '../store/chatStore'
import { Sparkles, Scale3D, LineChart } from 'lucide-react'

interface Tab {
  key: TabKey
  label: string
  icon: React.ReactNode
}

const tabs: Tab[] = [
  { key: 'fund-qa', label: '基金问答', icon: <Scale3D size={16} /> },
  { key: 'research', label: '投研分析', icon: <LineChart size={16} /> },
  { key: 'wealth-advisor', label: '财富顾问', icon: <Sparkles size={16} /> },
]

export default function Header() {
  const activeTab = useChatStore((s) => s.activeTab)
  const setActiveTab = useChatStore((s) => s.setActiveTab)

  return (
    <header className="sticky top-0 z-50 border-b border-slate-700/50 bg-slate-900/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-3xl items-center justify-between px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-amber-400 to-amber-600">
            <Sparkles size={16} className="text-slate-900" />
          </div>
          <h1 className="text-base font-semibold text-slate-100">
            智能投顾AI助手
          </h1>
        </div>

        <nav className="flex rounded-xl bg-slate-800 p-0.5" role="tablist">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.key
            return (
              <button
                key={tab.key}
                role="tab"
                aria-selected={isActive}
                onClick={() => setActiveTab(tab.key)}
                className={`
                  flex items-center gap-1.5 rounded-[10px] px-3 py-1.5 text-xs font-medium
                  transition-all duration-200
                  ${
                    isActive
                      ? 'bg-amber-500/20 text-amber-300 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }
                `}
              >
                {tab.icon}
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            )
          })}
        </nav>
      </div>
    </header>
  )
}