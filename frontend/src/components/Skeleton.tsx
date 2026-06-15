/**
 * 骨架屏组件 - 用于加载状态的占位动画
 */

interface SkeletonProps {
  className?: string
  /** 预设尺寸 */
  variant?: 'text' | 'title' | 'card' | 'circle' | 'button'
}

const variantClasses: Record<string, string> = {
  text: 'h-3 w-full',
  title: 'h-5 w-3/4',
  card: 'h-24 w-full rounded-xl',
  circle: 'h-10 w-10 rounded-full',
  button: 'h-9 w-20 rounded-lg',
}

export function Skeleton({ className = '', variant = 'text' }: SkeletonProps) {
  return (
    <div
      className={`skeleton ${variantClasses[variant] || ''} ${className}`}
      aria-hidden="true"
    />
  )
}

/** 消息加载骨架屏 */
export function MessageSkeleton() {
  return (
    <div className="message-animate flex justify-start">
      <div className="glass-card max-w-[80%] rounded-2xl px-5 py-4 space-y-3">
        <Skeleton variant="text" className="w-20" />
        <Skeleton variant="text" className="w-full" />
        <Skeleton variant="text" className="w-4/5" />
        <Skeleton variant="text" className="w-3/5" />
      </div>
    </div>
  )
}

/** 报告加载骨架屏 */
export function ReportSkeleton() {
  return (
    <div className="message-animate flex justify-start">
      <div className="glass-card max-w-[85%] rounded-2xl px-5 py-4 space-y-3 w-full">
        <Skeleton variant="title" />
        <Skeleton variant="text" />
        <Skeleton variant="text" />
        <Skeleton variant="card" />
        <Skeleton variant="text" className="w-1/2" />
        <Skeleton variant="text" />
        <Skeleton variant="text" className="w-3/4" />
      </div>
    </div>
  )
}