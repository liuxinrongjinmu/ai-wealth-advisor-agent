import { useState, useEffect } from 'react'
import { X, MessageSquare, Clock, Trash2, ChevronRight } from 'lucide-react'

interface ThreadSummary {
  thread_id: string
  customer_id: string
  tab_type: string
  started_at: string
  last_active: string
  message_count: number
  /** 第一条用户消息，用作对话标题 */
  first_message?: string
}

interface Props {
  isOpen: boolean
  onClose: () => void
  /** 点击加载历史对话，传入 threadId 和 tabType */
  onSelectThread?: (threadId: string, tabType: string) => void
}

/**
 * 对话历史侧边栏组件
 * 浏览和选择历史对话线程
 */
export default function ConversationHistory({ isOpen, onClose, onSelectThread }: Props) {
  const [threads, setThreads] = useState<ThreadSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (isOpen) {
      fetchThreads()
    }
  }, [isOpen])

  const fetchThreads = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await fetch('/api/v1/conversations')
      if (res.ok) {
        const data = await res.json()
        setThreads(data)
      } else {
        setError('获取对话历史失败')
      }
    } catch {
      setError('网络错误，请稍后重试')
    } finally {
      setLoading(false)
    }
  }

  /** 删除指定对话线程 */
  const handleDelete = async (e: React.MouseEvent, threadId: string) => {
    e.stopPropagation() // 阻止触发 onSelectThread
    try {
      const res = await fetch(`/api/v1/conversations/${threadId}`, { method: 'DELETE' })
      if (res.ok) {
        setThreads((prev) => prev.filter((t) => t.thread_id !== threadId))
      }
    } catch {
      // 静默失败
    }
  }

  const tabLabel = (type: string) => {
    switch (type) {
      case 'fund-qa': return '基金问答'
      case 'research': return '投研分析'
      case 'wealth-advisor': return '财富顾问'
      default: return type
    }
  }

  const formatTime = (ts: string) => {
    try {
      const d = new Date(ts)
      const now = new Date()
      const diff = now.getTime() - d.getTime()
      if (diff < 60 * 60 * 1000) return `${Math.floor(diff / 60000)}分钟前`
      if (diff < 24 * 60 * 60 * 1000) return `${Math.floor(diff / 3600000)}小时前`
      return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    } catch {
      return ts
    }
  }

  const getPreview = (thread: ThreadSummary): string => {
    // 使用第一条用户消息作为对话标题（自动生成）
    if (thread.first_message) {
      return thread.first_message.length > 30
        ? thread.first_message.slice(0, 30) + '...'
        : thread.first_message
    }
    if (thread.tab_type === 'fund-qa') return '私募基金法规问答'
    if (thread.tab_type === 'research') return '投研分析报告'
    return '财富管理咨询'
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-y-0 left-0 z-40 w-72 glass-card shadow-2xl border-r border-border flex flex-col">
      {/* 头部 */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-4 w-4 text-accent" />
          <h2 className="text-sm font-semibold text-text-primary">对话历史</h2>
        </div>
        <button
          onClick={onClose}
          className="rounded-lg p-1.5 text-text-muted hover:text-text-primary hover:bg-bg-tertiary transition-colors"
          aria-label="关闭历史"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* 内容 */}
      <div className="flex-1 overflow-y-auto py-2">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="skeleton h-4 w-24 rounded" />
          </div>
        ) : error ? (
          <div className="px-4 py-4 text-xs text-text-muted text-center">{error}</div>
        ) : threads.length === 0 ? (
          <div className="px-4 py-8 text-center">
            <Clock className="h-8 w-8 mx-auto mb-2 text-text-muted opacity-40" />
            <p className="text-xs text-text-muted">暂无对话历史</p>
            <p className="text-[10px] text-text-muted mt-1">开始与AI对话后将在此显示</p>
          </div>
        ) : (
          threads.map((thread) => (
            <button
              key={thread.thread_id}
              onClick={() => onSelectThread?.(thread.thread_id, thread.tab_type)}
              className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-bg-tertiary/50 transition-colors text-left group"
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5 mb-0.5">
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent/10 text-accent font-medium">
                    {tabLabel(thread.tab_type)}
                  </span>
                  <span className="text-[10px] text-text-muted">
                    {thread.message_count}条消息
                  </span>
                </div>
                <p className="text-xs text-text-secondary truncate">
                  {getPreview(thread)}
                </p>
                <p className="text-[10px] text-text-muted mt-0.5">
                  {formatTime(thread.last_active)}
                </p>
              </div>
              <ChevronRight className="h-3.5 w-3.5 text-text-muted opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
              <button
                onClick={(e) => handleDelete(e, thread.thread_id)}
                className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-red-50 dark:hover:bg-red-900/20"
                aria-label="删除对话"
                title="删除对话"
              >
                <Trash2 className="h-3.5 w-3.5 text-red-400 hover:text-red-500" />
              </button>
            </button>
          ))
        )}
      </div>

      {/* 底部操作 */}
      <div className="px-4 py-3 border-t border-border">
        <button
          onClick={fetchThreads}
          className="w-full glass-light rounded-lg px-3 py-1.5 text-xs text-text-secondary hover:text-text-primary transition-colors"
        >
          刷新列表
        </button>
      </div>
    </div>
  )
}