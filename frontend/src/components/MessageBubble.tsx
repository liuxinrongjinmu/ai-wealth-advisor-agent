import { useState } from 'react'
import { Copy, Check, Download, RotateCcw, ThumbsUp, ThumbsDown } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useChatStore, type Message } from '../store'

interface Props {
  message: Message
  onRegenerate?: () => void
  showRegenerate?: boolean
}

export default function MessageBubble({ message, onRegenerate, showRegenerate }: Props) {
  const isUser = message.role === 'user'
  const [copied, setCopied] = useState(false)
  const setMessageFeedback = useChatStore((s) => s.setMessageFeedback)
  const addToast = useChatStore((s) => s.addToast)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
    addToast({ type: 'success', message: '已复制到剪贴板', duration: 2000 })
  }

  const handleExport = () => {
    const blob = new Blob([message.content], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `research-report-${new Date(message.timestamp).toISOString().slice(0, 10)}.md`
    a.click()
    URL.revokeObjectURL(url)
    addToast({ type: 'success', message: '报告已导出', duration: 2000 })
  }

  const handleFeedback = (feedback: 1 | -1) => {
    const newVal = message.feedback === feedback ? 0 : feedback
    setMessageFeedback(message.id, newVal)
    if (newVal === 1) {
      addToast({ type: 'success', message: '感谢您的反馈！', duration: 1500 })
    }
  }

  const isLongReport = !isUser && message.content.length > 500
  const genTime = new Date(message.timestamp).toLocaleString('zh-CN', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })

  return (
    <div className={`message-animate flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className="relative group max-w-[85%]">
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? 'bg-gradient-to-br from-accent to-accent-dark text-white shadow-lg shadow-accent/15 whitespace-pre-wrap'
              : 'glass-card text-text-primary prose prose-sm max-w-none prose-headings:text-text-primary prose-p:text-text-secondary prose-strong:text-text-primary prose-a:text-accent prose-code:bg-bg-tertiary prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs prose-pre:bg-bg-secondary prose-pre:border prose-pre:border-border-light prose-table:border-collapse'
          }`}
        >
          {!isUser && (
            <div className="mb-2 flex items-center gap-1.5">
              <div className="h-1.5 w-1.5 rounded-full bg-accent float-glow" />
              <span className="text-[10px] font-medium uppercase tracking-widest text-accent">
                AI Assistant
              </span>
            </div>
          )}
          {isUser ? (
            message.content
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          )}
          {/* 生成时间戳 */}
          {!isUser && (
            <div className="mt-2 text-[10px] text-text-muted text-right">
              生成于 {genTime}
            </div>
          )}
        </div>

        {/* 操作按钮（非用户消息悬停显示） */}
        {!isUser && (
          <div className="absolute top-1 right-1 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            {showRegenerate && onRegenerate && (
              <button
                onClick={onRegenerate}
                className="glass-light rounded-md p-1 hover:bg-white/60 transition-colors"
                title="重新生成"
              >
                <RotateCcw className="h-3.5 w-3.5 text-text-secondary" />
              </button>
            )}
            <button
              onClick={handleCopy}
              className="glass-light rounded-md p-1 hover:bg-white/60 transition-colors"
              title="复制"
            >
              {copied ? (
                <Check className="h-3.5 w-3.5 text-success" />
              ) : (
                <Copy className="h-3.5 w-3.5 text-text-secondary" />
              )}
            </button>
            {isLongReport && (
              <button
                onClick={handleExport}
                className="glass-light rounded-md p-1 hover:bg-white/60 transition-colors"
                title="导出Markdown"
              >
                <Download className="h-3.5 w-3.5 text-text-secondary" />
              </button>
            )}
          </div>
        )}

        {/* 反馈按钮（非用户消息底部显示） */}
        {!isUser && (
          <div className="mt-1.5 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            <button
              onClick={() => handleFeedback(1)}
              className={`rounded-md p-1 transition-colors ${
                message.feedback === 1
                  ? 'text-green-500 bg-green-50'
                  : 'text-text-muted hover:text-green-500 hover:bg-green-50/50'
              }`}
              title="有帮助"
            >
              <ThumbsUp className="h-3 w-3" />
            </button>
            <button
              onClick={() => handleFeedback(-1)}
              className={`rounded-md p-1 transition-colors ${
                message.feedback === -1
                  ? 'text-red-500 bg-red-50'
                  : 'text-text-muted hover:text-red-500 hover:bg-red-50/50'
              }`}
              title="无帮助"
            >
              <ThumbsDown className="h-3 w-3" />
            </button>
          </div>
        )}
      </div>
    </div>
  )
}