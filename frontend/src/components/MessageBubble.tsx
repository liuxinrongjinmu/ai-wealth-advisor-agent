import type { Message } from '../store'

interface Props {
  message: Message
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'

  return (
    <div className={`message-animate flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? 'bg-gradient-to-br from-accent to-accent-dark text-white shadow-lg shadow-accent/15'
            : 'glass-card text-text-primary'
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
        {message.content}
      </div>
    </div>
  )
}
