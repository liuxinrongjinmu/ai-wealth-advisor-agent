import type { Message } from '../store/chatStore'
import { Bot, User } from 'lucide-react'

interface Props {
  message: Message
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'

  return (
    <div
      className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'} animate-in fade-in slide-in-from-bottom-2 duration-300`}
    >
      {/* 头像 */}
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
          isUser
            ? 'bg-gradient-to-br from-emerald-400 to-teal-500'
            : 'bg-gradient-to-br from-amber-400 to-amber-600'
        }`}
      >
        {isUser ? (
          <User size={14} className="text-white" />
        ) : (
          <Bot size={14} className="text-slate-900" />
        )}
      </div>

      {/* 消息内容 */}
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? 'bg-gradient-to-br from-emerald-500 to-teal-600 text-white shadow-lg shadow-emerald-500/20 rounded-br-md'
            : 'bg-slate-800 text-slate-200 shadow-lg shadow-slate-900/30 rounded-bl-md border border-slate-700/50'
        }`}
      >
        <div className="whitespace-pre-wrap break-words">{message.content}</div>
      </div>
    </div>
  )
}