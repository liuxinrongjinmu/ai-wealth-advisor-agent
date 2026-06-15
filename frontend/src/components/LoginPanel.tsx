import { useState } from 'react'
import { X, User, Mail, Lock, Loader2 } from 'lucide-react'
import { login, register, type AuthUser, refreshAccessToken } from '../api'

interface Props {
  isOpen: boolean
  onClose: () => void
  onAuthSuccess: (user: AuthUser) => void
}

/**
 * 登录/注册面板组件
 * 支持登录和注册双模式切换
 */
export default function LoginPanel({ isOpen, onClose, onAuthSuccess }: Props) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  if (!isOpen) return null

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      let result
      if (mode === 'login') {
        result = await login(username.trim(), password)
      } else {
        result = await register(username.trim(), password, displayName.trim())
      }
      onAuthSuccess(result.user)
      setUsername('')
      setPassword('')
      setDisplayName('')
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : '操作失败，请重试')
    } finally {
      setLoading(false)
    }
  }

  /** 尝试使用refresh token自动登录 */
  const tryAutoLogin = async () => {
    const refreshed = await refreshAccessToken()
    if (refreshed) {
      onAuthSuccess(refreshed.user)
      onClose()
      return
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/20 backdrop-blur-sm">
      <div className="glass-card w-full max-w-sm mx-4 rounded-2xl shadow-2xl">
        {/* 头部 */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <h2 className="text-sm font-semibold text-text-primary">
            {mode === 'login' ? '登录' : '注册'}
          </h2>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-text-muted hover:text-text-primary hover:bg-bg-tertiary transition-colors"
            aria-label="关闭"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* 表单 */}
        <form onSubmit={handleSubmit} className="px-5 py-4 space-y-3">
          {error && (
            <div className="rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-xs text-red-700">
              {error}
            </div>
          )}

          <div>
            <label className="flex items-center gap-1.5 mb-1.5 text-xs font-medium text-text-secondary">
              <User className="h-3 w-3" />
              用户名
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="请输入用户名（至少3位）"
              className="w-full glass rounded-lg border border-border px-3 py-2 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50 transition-colors"
              required
              minLength={3}
              maxLength={30}
            />
          </div>

          {mode === 'register' && (
            <div>
              <label className="flex items-center gap-1.5 mb-1.5 text-xs font-medium text-text-secondary">
                <Mail className="h-3 w-3" />
                显示名称
              </label>
              <input
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="选填，默认为用户名"
                className="w-full glass rounded-lg border border-border px-3 py-2 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50 transition-colors"
                maxLength={50}
              />
            </div>
          )}

          <div>
            <label className="flex items-center gap-1.5 mb-1.5 text-xs font-medium text-text-secondary">
              <Lock className="h-3 w-3" />
              密码
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="请输入密码（至少6位）"
              className="w-full glass rounded-lg border border-border px-3 py-2 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent/50 transition-colors"
              required
              minLength={6}
              maxLength={50}
            />
          </div>

          <button
            type="submit"
            disabled={loading || !username.trim() || !password}
            className="w-full flex items-center justify-center gap-2 rounded-lg bg-accent px-4 py-2.5 text-xs font-medium text-white hover:bg-accent-dark disabled:opacity-50 transition-all"
          >
            {loading ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                {mode === 'login' ? '登录中...' : '注册中...'}
              </>
            ) : mode === 'login' ? (
              '登录'
            ) : (
              '注册'
            )}
          </button>

          <div className="text-center">
            <button
              type="button"
              onClick={() => {
                setMode(mode === 'login' ? 'register' : 'login')
                setError('')
              }}
              className="text-xs text-accent hover:underline"
            >
              {mode === 'login' ? '没有账号？立即注册' : '已有账号？立即登录'}
            </button>
            {' | '}
            <button
              type="button"
              onClick={tryAutoLogin}
              className="text-xs text-text-muted hover:text-text-secondary"
            >
              自动登录
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}