import { useState, useRef, useEffect, type KeyboardEvent } from 'react'
import { Send, Loader2 } from 'lucide-react'

interface Props {
  onSend: (query: string) => void
  isLoading: boolean
  placeholder?: string
  value: string
  onChange: (v: string) => void
}

export default function InputArea({
  onSend,
  isLoading,
  placeholder = '输入您的问题...',
  value,
  onChange,
}: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`
    }
  }, [value])

  const handleSend = () => {
    const trimmed = value.trim()
    if (!trimmed || isLoading) return
    onSend(trimmed)
    onChange('')
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="sticky bottom-0 border-t border-slate-700/50 bg-slate-900/90 backdrop-blur-xl">
      <div className="mx-auto max-w-3xl px-4 py-3">
        <div className="flex items-end gap-2 rounded-2xl bg-slate-800 p-2 ring-1 ring-slate-700/50 focus-within:ring-amber-500/50 transition-all duration-300">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={isLoading}
            rows={1}
            className="flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-slate-100 placeholder-slate-500 outline-none
                       disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={!value.trim() || isLoading}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl
                       bg-gradient-to-br from-amber-400 to-amber-600 text-slate-900
                       transition-all duration-200 hover:from-amber-300 hover:to-amber-500
                       disabled:opacity-40 disabled:cursor-not-allowed active:scale-95"
            aria-label="发送"
          >
            {isLoading ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Send size={16} />
            )}
          </button>
        </div>
        <p className="mt-1.5 text-center text-[10px] text-slate-600">
          Enter 发送 · Shift+Enter 换行
        </p>
      </div>
    </div>
  )
}