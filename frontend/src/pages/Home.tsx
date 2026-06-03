import { useState, useRef, useEffect } from 'react'
import { useChatStore } from '../store/chatStore'
import { askFundQA, askResearch, askWealthAdvisor } from '../api'
import Header from '../components/Header'
import MessageBubble from '../components/MessageBubble'
import QuickPrompts from '../components/QuickPrompts'
import InputArea from '../components/InputArea'
import { Bot } from 'lucide-react'

export default function Home() {
  const messages = useChatStore((s) => s.messages)
  const activeTab = useChatStore((s) => s.activeTab)
  const isLoading = useChatStore((s) => s.isLoading)
  const setLoading = useChatStore((s) => s.setLoading)
  const addMessage = useChatStore((s) => s.addMessage)
  const customerId = useChatStore((s) => s.customerId)

  const [inputValue, setInputValue] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const generateId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

  const handleSend = async (query: string) => {
    const userMsg = {
      id: generateId(),
      role: 'user' as const,
      content: query,
      timestamp: Date.now(),
    }
    addMessage(userMsg)
    setLoading(true)

    try {
      let response = ''

      if (activeTab === 'fund-qa') {
        const res = await askFundQA(query)
        response = res.answer
      } else if (activeTab === 'research') {
        const res = await askResearch(query)
        if (res.error) {
          response = `分析出错: ${res.error}`
        } else {
          response = res.report || '未生成报告'
        }
      } else if (activeTab === 'wealth-advisor') {
        const res = await askWealthAdvisor(query, customerId)
        if (res.error) {
          response = `处理出错: ${res.error}`
        } else {
          response = res.response || '未生成回复'
        }
      }

      addMessage({
        id: generateId(),
        role: 'assistant',
        content: response,
        timestamp: Date.now(),
      })
    } catch (err) {
      addMessage({
        id: generateId(),
        role: 'assistant',
        content: `请求失败: ${err instanceof Error ? err.message : '未知错误'}`,
        timestamp: Date.now(),
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-screen flex-col bg-slate-950">
      <Header />

      {/* 消息列表 */}
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-4 py-6">
          {messages.length === 0 ? (
            /* 空状态欢迎 */
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-400/20 to-amber-600/20 ring-1 ring-amber-500/30">
                <Bot size={32} className="text-amber-400" />
              </div>
              <h2 className="mb-2 text-lg font-semibold text-slate-200">
                智能投顾AI助手
              </h2>
              <p className="mb-6 max-w-sm text-sm text-slate-500">
                选择一个子系统开始对话，输入您的问题获取专业的金融咨询服务
              </p>
              <QuickPrompts onSelect={setInputValue} />
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              {messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))}
              {isLoading && (
                <div className="flex items-center gap-3">
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-amber-400 to-amber-600">
                    <Bot size={14} className="text-slate-900" />
                  </div>
                  <div className="flex items-center gap-1 rounded-2xl rounded-bl-md bg-slate-800 px-4 py-3">
                    <span className="h-2 w-2 animate-bounce rounded-full bg-amber-400 [animation-delay:0ms]" />
                    <span className="h-2 w-2 animate-bounce rounded-full bg-amber-400 [animation-delay:150ms]" />
                    <span className="h-2 w-2 animate-bounce rounded-full bg-amber-400 [animation-delay:300ms]" />
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
      </main>

      {/* 快捷提问（有消息时在输入框上方显示） */}
      {messages.length > 0 && (
        <div className="mx-auto w-full max-w-3xl px-4 pb-1">
          <QuickPrompts onSelect={setInputValue} />
        </div>
      )}

      <InputArea
        onSend={handleSend}
        isLoading={isLoading}
        value={inputValue}
        onChange={setInputValue}
      />
    </div>
  )
}