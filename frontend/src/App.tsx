import { useCallback, useRef, useState, useMemo, useEffect } from 'react'
import Header from './components/Header'
import ChatArea from './components/ChatArea'
import InputArea from './components/InputArea'
import ToastContainer from './components/Toast'
import SettingsPanel from './components/SettingsPanel'
import LoginPanel from './components/LoginPanel'
import ConversationHistory from './components/ConversationHistory'
import { useChatStore, type ResearchStage } from './store'
import { fundQA, researchStream, wealthAdvisor, getCurrentUser, loadThreadMessages, type AuthUser } from './api'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'

const ALL_STAGES: ResearchStage[] = [
  { key: 'perception', name: '感知阶段', status: 'pending', message: '等待开始' },
  { key: 'modeling', name: '建模阶段', status: 'pending', message: '等待开始' },
  { key: 'reasoning', name: '推理阶段', status: 'pending', message: '等待开始' },
  { key: 'decision', name: '决策阶段', status: 'pending', message: '等待开始' },
  { key: 'report', name: '报告阶段', status: 'pending', message: '等待开始' },
]

export default function App() {
  const {
    activeTab, customerId, apiKey, threadId, messages,
    addMessage, setLoading, setStreaming,
    setResearchStages, updateResearchStage,
    removeLastAssistantMessage, setActiveTab,
    toasts, addToast, removeToast,
    authUser, setAuthUser,
  } = useChatStore()
  const abortRef = useRef<AbortController | null>(null)
  const [researchIndustry, setResearchIndustry] = useState('综合')
  const [researchHorizon, setResearchHorizon] = useState('中期')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [loginOpen, setLoginOpen] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)

  // 自动尝试登录（仅组件挂载时执行一次，避免依赖循环）
  useEffect(() => {
    const stored = localStorage.getItem('wealth-advisor-state')
    if (stored) {
      try {
        const parsed = JSON.parse(stored)
        if (parsed.authUser) {
          setAuthUser(parsed.authUser)
          return
        }
      } catch { /* ignore */ }
    }
    // 无本地存储时尝试refresh token
    getCurrentUser().then((user) => {
      if (user) setAuthUser(user)
    })
  }, []) // 修复：移除 authUser 依赖，避免循环触发

  // 认证成功回调
  const handleAuthSuccess = useCallback((user: AuthUser) => {
    setAuthUser(user)
    addToast({ type: 'success', message: `欢迎回来，${user.display_name || user.username}！`, duration: 3000 })
  }, [setAuthUser, addToast])

  // 核心发送逻辑
  const executeSend = useCallback(
    async (message: string) => {
      if (abortRef.current) {
        abortRef.current.abort()
      }
      const controller = new AbortController()
      abortRef.current = controller

      setLoading(true)

      try {
        let response = ''

        if (activeTab === 'fund-qa') {
          const res = await fundQA(message, threadId, apiKey || undefined, controller.signal)
          response = res.answer
        } else if (activeTab === 'research') {
          setStreaming(true)
          setResearchStages(ALL_STAGES.map((s) => ({ ...s, status: 'pending' as const, message: '等待开始' })))

          try {
            await researchStream(
              message, researchIndustry, researchHorizon, threadId, apiKey || undefined,
              (stage, name, msg) => {
                updateResearchStage(stage, { status: 'running', name, message: msg })
              },
              (stage, name, summary) => {
                updateResearchStage(stage, { status: 'done', name, message: summary })
              },
              (report) => {
                response = report
              },
              (error) => {
                response = `流式分析出错：${error}`
              },
              controller.signal,
            )
          } finally {
            setStreaming(false)
          }
        } else {
          const res = await wealthAdvisor(message, customerId, threadId, apiKey || undefined, controller.signal)
          response = res.error
            ? `处理出错：${res.error}`
            : `${res.processing_mode === 'reactive' ? '[快速响应]' : '[深度分析]'} ${res.response}`
        }

        if (response) {
          addMessage({ role: 'assistant', content: response })
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          addToast({ type: 'info', message: '已停止生成' })
          return
        }
        const errorMsg = err instanceof Error ? err.message : '未知错误'
        // 友好错误提示
        let friendlyMsg = `请求失败：${errorMsg}`
        if (errorMsg.includes('超时') || errorMsg.includes('504')) {
          friendlyMsg = '请求超时，请稍后重试或简化问题'
        } else if (errorMsg.includes('500') || errorMsg.includes('出错')) {
          friendlyMsg = '服务暂时不可用，请稍后重试'
        } else if (errorMsg.includes('429')) {
          friendlyMsg = '请求过于频繁，请稍后再试'
        }
        addMessage({
          role: 'assistant',
          content: friendlyMsg,
        })
        addToast({ type: 'error', message: `请求失败：${errorMsg}`, duration: 5000 })
      } finally {
        setLoading(false)
        abortRef.current = null
      }
    },
    [activeTab, customerId, apiKey, threadId, researchIndustry, researchHorizon,
     addMessage, setLoading, setStreaming, setResearchStages, updateResearchStage],
  )

  // 发送新消息
  const handleSend = useCallback(
    (message: string) => {
      addMessage({ role: 'user', content: message })
      executeSend(message)
    },
    [addMessage, executeSend],
  )

  // 停止生成
  const handleStop = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
    setLoading(false)
    setStreaming(false)
  }, [setLoading, setStreaming])

  // 重新生成
  const handleRegenerate = useCallback(() => {
    const lastUserQuery = removeLastAssistantMessage()
    if (lastUserQuery) {
      executeSend(lastUserQuery)
    }
  }, [removeLastAssistantMessage, executeSend])

  // 加载历史对话线程
  const handleSelectThread = useCallback(
    async (threadId: string, tabType: string) => {
      try {
        const msgs = await loadThreadMessages(threadId)
        // 切换到对应Tab
        if (tabType === 'fund-qa' || tabType === 'research' || tabType === 'wealth-advisor') {
          setActiveTab(tabType)
        }
        // 加载消息（带ID和时间戳）
        const loadedMessages = msgs.map((m: any) => ({
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
          role: m.role as 'user' | 'assistant',
          content: 'content' in m ? m.content : (m.answer || m.report || m.response || ''),
          timestamp: Date.now(),
        }))
        // 设置消息和线程ID
        useChatStore.setState({
          messages: loadedMessages,
          threadId,
        })
        addToast({ type: 'success', message: `已加载 ${loadedMessages.length} 条历史消息`, duration: 2000 })
      } catch (err) {
        addToast({ type: 'error', message: '加载历史对话失败', duration: 3000 })
      }
    },
    [setActiveTab, addToast],
  )

  // 键盘快捷键
  const shortcuts = useMemo(() => [
    {
      keys: 'escape',
      description: '停止生成 / 取消操作',
      handler: handleStop,
    },
    {
      keys: 'ctrl+enter',
      description: '重新生成最后一条回答',
      handler: handleRegenerate,
    },
  ], [handleStop, handleRegenerate])
  useKeyboardShortcuts(shortcuts)

  return (
    <div className="flex h-screen flex-col bg-bg-primary">
      <Header
        onOpenSettings={() => setSettingsOpen(true)}
        onOpenLogin={() => setLoginOpen(true)}
        onOpenHistory={() => setHistoryOpen(true)}
      />
      <ChatArea
        researchIndustry={researchIndustry}
        researchHorizon={researchHorizon}
        onIndustryChange={setResearchIndustry}
        onHorizonChange={setResearchHorizon}
        onRegenerate={handleRegenerate}
        onSendPrompt={handleSend}
      />
      <InputArea
        onSend={handleSend}
        onStop={handleStop}
        researchIndustry={researchIndustry}
        researchHorizon={researchHorizon}
        onIndustryChange={setResearchIndustry}
        onHorizonChange={setResearchHorizon}
      />
      <ConversationHistory
        isOpen={historyOpen}
        onClose={() => setHistoryOpen(false)}
        onSelectThread={handleSelectThread}
      />
      <SettingsPanel isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <LoginPanel
        isOpen={loginOpen}
        onClose={() => setLoginOpen(false)}
        onAuthSuccess={handleAuthSuccess}
      />
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </div>
  )
}