import { useEffect, useState, useCallback } from 'react'
import { X, CheckCircle, AlertCircle, Info, AlertTriangle } from 'lucide-react'

export type ToastType = 'success' | 'error' | 'info' | 'warning'

export interface ToastItem {
  id: string
  type: ToastType
  message: string
  duration?: number
}

interface ToastProps {
  toasts: ToastItem[]
  onRemove: (id: string) => void
}

const iconMap = {
  success: CheckCircle,
  error: AlertCircle,
  info: Info,
  warning: AlertTriangle,
}

const colorMap = {
  success: 'border-green-400 bg-green-50 text-green-800',
  error: 'border-red-400 bg-red-50 text-red-800',
  info: 'border-blue-400 bg-blue-50 text-blue-800',
  warning: 'border-yellow-400 bg-yellow-50 text-yellow-800',
}

const iconColorMap = {
  success: 'text-green-500',
  error: 'text-red-500',
  info: 'text-blue-500',
  warning: 'text-yellow-500',
}

function ToastItemView({ toast, onRemove }: { toast: ToastItem; onRemove: () => void }) {
  useEffect(() => {
    const duration = toast.duration || 4000
    const timer = setTimeout(onRemove, duration)
    return () => clearTimeout(timer)
  }, [toast.duration, onRemove])

  const Icon = iconMap[toast.type]

  return (
    <div
      className={`toast-animate glass-card flex items-start gap-2.5 rounded-xl border px-4 py-3 shadow-lg ${colorMap[toast.type]}`}
      role="alert"
    >
      <Icon className={`h-4 w-4 shrink-0 mt-0.5 ${iconColorMap[toast.type]}`} />
      <p className="flex-1 text-xs leading-relaxed">{toast.message}</p>
      <button
        onClick={onRemove}
        className="shrink-0 rounded-md p-0.5 hover:bg-black/5 transition-colors"
        aria-label="关闭通知"
      >
        <X className="h-3.5 w-3.5 opacity-60" />
      </button>
    </div>
  )
}

export default function ToastContainer({ toasts, onRemove }: ToastProps) {
  if (toasts.length === 0) return null

  return (
    <div className="fixed bottom-20 right-4 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none">
      {toasts.map((toast) => (
        <div key={toast.id} className="pointer-events-auto">
          <ToastItemView toast={toast} onRemove={() => onRemove(toast.id)} />
        </div>
      ))}
    </div>
  )
}