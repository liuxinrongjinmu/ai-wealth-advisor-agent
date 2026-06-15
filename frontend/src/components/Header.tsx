import { Cpu, Plus, Sun, Moon, Settings, LogIn, User, History } from 'lucide-react'
import { useChatStore, type TabType } from '../store'
import { logout } from '../api'

const tabs: { key: TabType; label: string; icon: string }[] = [
  { key: 'fund-qa', label: '基金问答', icon: '📋' },
  { key: 'research', label: '投研分析', icon: '🔬' },
  { key: 'wealth-advisor', label: '财富顾问', icon: '💎' },
]

interface Props {
  onOpenSettings: () => void
  onOpenLogin: () => void
  onOpenHistory: () => void
}

export default function Header({ onOpenSettings, onOpenLogin, onOpenHistory }: Props) {
  const { activeTab, setActiveTab, newThread, messages, theme, toggleTheme, authUser, setAuthUser } = useChatStore()

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

        <div className="flex items-center gap-2">
          {/* 导航标签 */}
          <nav className="glass-light hidden sm:flex gap-1 rounded-xl p-1">
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

          {/* 移动端Tab下拉 */}
          <select
            value={activeTab}
            onChange={(e) => setActiveTab(e.target.value as TabType)}
            className="sm:hidden glass-light rounded-lg px-2.5 py-1.5 text-xs text-text-primary border border-border focus:outline-none"
          >
            {tabs.map((t) => (
              <option key={t.key} value={t.key}>{t.icon} {t.label}</option>
            ))}
          </select>

          {/* 新对话按钮 */}
          {messages.length > 0 && (
            <button
              onClick={newThread}
              className="glass-light flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs text-text-secondary hover:text-accent transition-colors"
              title="开始新对话"
            >
              <Plus className="h-3.5 w-3.5" />
            </button>
          )}

          {/* 右侧功能区 */}
          <div className="flex items-center gap-1.5">
            {/* 对话历史 */}
            <button
              onClick={onOpenHistory}
              className="glass-light flex items-center justify-center rounded-lg w-8 h-8 text-text-secondary hover:text-text-primary transition-colors"
              title="对话历史"
            >
              <History className="h-4 w-4" />
            </button>

            {/* 暗色模式切换 */}
            <button
              onClick={toggleTheme}
              className="glass-light flex items-center justify-center rounded-lg w-8 h-8 text-text-secondary hover:text-text-primary transition-colors"
              title={theme === 'light' ? '切换暗色模式' : '切换亮色模式'}
            >
              {theme === 'light' ? (
                <Moon className="h-4 w-4" />
              ) : (
                <Sun className="h-4 w-4" />
              )}
            </button>

            {/* 设置按钮 */}
            <button
              onClick={onOpenSettings}
              className="glass-light flex items-center justify-center rounded-lg w-8 h-8 text-text-secondary hover:text-text-primary transition-colors"
              title="设置"
            >
              <Settings className="h-4 w-4" />
            </button>

            {/* 用户状态 */}
            {authUser ? (
              <button
                onClick={() => {
                  logout()
                  setAuthUser(null)
                }}
                className="glass-light flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs text-text-secondary hover:text-text-primary transition-colors"
                title={`当前用户: ${authUser.display_name || authUser.username}（点击退出）`}
              >
                <User className="h-3.5 w-3.5" />
                <span className="max-w-[60px] truncate">{authUser.display_name || authUser.username}</span>
              </button>
            ) : (
              <button
                onClick={onOpenLogin}
                className="glass-light flex items-center justify-center rounded-lg w-8 h-8 text-text-secondary hover:text-accent transition-colors"
                title="登录 / 注册"
              >
                <LogIn className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}