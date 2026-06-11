import { useCallback, useRef, useState } from 'react'
import Header from './components/Header'
import ChatArea from './components/ChatArea'
import InputArea from './components/InputArea'
import { useChatStore } from './store'
import { fundQA, research, wealthAdvisor } from './api'

export default function App() {
  const { activeTab, customerId, apiKey, addMessage, setLoading } = useChatStore()
  const abortRef = useRef<AbortController | null>(null)
  const [researchIndustry, setResearchIndustry] = useState('综合')
  const [researchHorizon, setResearchHorizon] = useState('中期')

  const handleSend = useCallback(
    async (message: string) => {
      // 取消正在进行的请求
      if (abortRef.current) {
        abortRef.current.abort()
      }
      const controller = new AbortController()
      abortRef.current = controller

      addMessage({ role: 'user', content: message })
      setLoading(true)

      try {
        let response = ''

        if (activeTab === 'fund-qa') {
          const res = await fundQA(message, apiKey || undefined, controller.signal)
          response = res.answer
        } else if (activeTab === 'research') {
          const res = await research(message, researchIndustry, researchHorizon, apiKey || undefined, controller.signal)
          response = res.error ? `分析出错：${res.error}` : res.report
        } else {
          const res = await wealthAdvisor(message, customerId, undefined, apiKey || undefined, controller.signal)
          response = res.error
            ? `处理出错：${res.error}`
            : `${res.processing_mode === 'reactive' ? '[快速响应]' : '[深度分析]'} ${res.response}`
        }

        addMessage({ role: 'assistant', content: response })
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          // 请求被取消，不显示错误
          return
        }
        addMessage({
          role: 'assistant',
          content: `请求失败：${err instanceof Error ? err.message : '未知错误'}`,
        })
      } finally {
        setLoading(false)
        abortRef.current = null
      }
    },
    [activeTab, customerId, apiKey, researchIndustry, researchHorizon, addMessage, setLoading]
  )

  return (
    <div className="flex h-screen flex-col bg-bg-primary">
      <Header />
      <ChatArea
        researchIndustry={researchIndustry}
        researchHorizon={researchHorizon}
        onIndustryChange={setResearchIndustry}
        onHorizonChange={setResearchHorizon}
      />
      <InputArea
        onSend={handleSend}
        researchIndustry={researchIndustry}
        researchHorizon={researchHorizon}
        onIndustryChange={setResearchIndustry}
        onHorizonChange={setResearchHorizon}
      />
    </div>
  )
}