import { Component, type ReactNode, type ErrorInfo } from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

/**
 * 错误边界组件
 * 捕获子组件渲染错误，防止整个应用崩溃
 */
export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('[ErrorBoundary] 捕获到错误:', error, errorInfo)
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback

      return (
        <div className="flex items-center justify-center min-h-[200px] p-8">
          <div className="glass-card rounded-2xl p-6 max-w-sm w-full text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-red-50">
              <AlertTriangle className="h-6 w-6 text-red-500" />
            </div>
            <h3 className="mb-2 text-sm font-semibold text-text-primary">页面出现异常</h3>
            <p className="mb-4 text-xs text-text-secondary leading-relaxed">
              {this.state.error?.message || '发生未知错误，请尝试刷新页面'}
            </p>
            <button
              onClick={this.handleReset}
              className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-4 py-2 text-xs text-white hover:bg-accent-dark transition-colors"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              重试
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}