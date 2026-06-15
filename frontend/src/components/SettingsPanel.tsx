import { useState } from 'react'
import { X, Key, Zap, Moon, Sun, Keyboard } from 'lucide-react'
import { useChatStore } from '../store'

interface Props {
  isOpen: boolean
  onClose: () => void
}

/**
 * 设置面板组件
 * 管理API Key、主题、快捷键提示等
 */
export default function SettingsPanel({ isOpen, onClose }: Props) {
  const { apiKey, setApiKey, theme, toggleTheme } = useChatStore()
  const [keyInput, setKeyInput] = useState(apiKey)
  const [saved, setSaved] = useState(false)

  if (!isOpen) return null

  const handleSave = () => {
    setApiKey(keyInput.trim())
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const shortcuts = [
    { keys: 'Esc', desc: '停止生成 / 取消操作' },
    { keys: 'Ctrl + Enter', desc: '重新生成最后一条回答' },
    { keys: 'Enter', desc: '发送消息' },
    { keys: 'Shift + Enter', desc: '换行' },
  ]

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/20 backdrop-blur-sm">
      <div className="glass-card w-full max-w-md mx-4 rounded-2xl shadow-2xl">
        {/* 头部 */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-accent" />
            <h2 className="text-sm font-semibold text-text-primary">设置</h2>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-text-muted hover:text-text-primary hover:bg-bg-tertiary transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* 内容 */}
        <div className="px-5 py-4 space-y-5">
          {/* API Key 设置 */}
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <Key className="h-3.5 w-3.5 text-text-muted" />
              <span className="text-xs font-medium text-text-secondary">LLM API Key</span>
            </div>
            <div className="flex gap-2">
              <input
                type="password"
                value={keyInput}
                onChange={(e) => setKeyInput(e.target.value)}
                placeholder="输入 DashScope API Key..."
                className="flex-1 glass rounded-lg border border-border px-3 py-2 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50 transition-colors"
              />
              <button
                onClick={handleSave}
                className={`rounded-lg px-3 py-2 text-xs font-medium transition-all ${
                  saved
                    ? 'bg-success text-white'
                    : 'bg-accent text-white hover:bg-accent-dark'
                }`}
              >
                {saved ? '已保存' : '保存'}
              </button>
            </div>
            <p className="mt-1.5 text-[10px] text-text-muted">
              不填写则使用后端环境变量中的 API Key
            </p>
          </div>

          {/* 主题切换 */}
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              {theme === 'light' ? (
                <Moon className="h-3.5 w-3.5 text-text-muted" />
              ) : (
                <Sun className="h-3.5 w-3.5 text-text-muted" />
              )}
              <span className="text-xs font-medium text-text-secondary">外观主题</span>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => theme !== 'light' && toggleTheme()}
                className={`flex-1 flex items-center justify-center gap-1.5 rounded-lg border px-3 py-2 text-xs transition-all ${
                  theme === 'light'
                    ? 'border-accent bg-accent/10 text-accent font-medium'
                    : 'border-border text-text-secondary hover:border-accent/30'
                }`}
              >
                <Sun className="h-3.5 w-3.5" />
                浅色
              </button>
              <button
                onClick={() => theme !== 'dark' && toggleTheme()}
                className={`flex-1 flex items-center justify-center gap-1.5 rounded-lg border px-3 py-2 text-xs transition-all ${
                  theme === 'dark'
                    ? 'border-accent bg-accent/10 text-accent font-medium'
                    : 'border-border text-text-secondary hover:border-accent/30'
                }`}
              >
                <Moon className="h-3.5 w-3.5" />
                深色
              </button>
            </div>
          </div>

          {/* 键盘快捷键 */}
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <Keyboard className="h-3.5 w-3.5 text-text-muted" />
              <span className="text-xs font-medium text-text-secondary">键盘快捷键</span>
            </div>
            <div className="glass rounded-xl border border-border divide-y divide-border/50">
              {shortcuts.map((s) => (
                <div key={s.keys} className="flex items-center justify-between px-3 py-2">
                  <span className="text-xs text-text-secondary">{s.desc}</span>
                  <kbd className="rounded-md bg-bg-tertiary px-2 py-0.5 text-[10px] font-mono text-text-muted border border-border">
                    {s.keys}
                  </kbd>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}