import { create } from 'zustand'
import type { ToastItem } from './components/Toast'
import type { AuthUser } from './api'

export type TabType = 'fund-qa' | 'research' | 'wealth-advisor'
export type ThemeType = 'light' | 'dark'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  /** 用户反馈：1=赞, -1=踩, 0=无 */
  feedback?: 1 | -1 | 0
}

/** 流式投研阶段的进度状态 */
export interface ResearchStage {
  key: string
  name: string
  status: 'pending' | 'running' | 'done'
  message: string
}

/** 从 localStorage 恢复的状态 */
function loadPersistedState() {
  try {
    const raw = localStorage.getItem('wealth-advisor-state')
    if (!raw) return {}
    const data = JSON.parse(raw)
    return {
      messages: data.messages || [],
      activeTab: data.activeTab || 'fund-qa',
      customerId: data.customerId || 'customer1',
      apiKey: data.apiKey || '',
      threadId: data.threadId || crypto.randomUUID(),
      theme: data.theme || 'light',
    }
  } catch { return {} }
}

interface ChatStore {
  messages: Message[]
  activeTab: TabType
  isLoading: boolean
  customerId: string
  apiKey: string
  researchStages: ResearchStage[]
  isStreaming: boolean
  threadId: string
  theme: ThemeType
  /** 当前登录用户 */
  authUser: AuthUser | null
  /** Toast通知列表 */
  toasts: ToastItem[]
  /** 每个Tab独立的消息缓存，切换Tab时保留对话 */
  tabMessages: Record<TabType, Message[]>
  /** 每个Tab独立的线程ID缓存 */
  tabThreadIds: Record<TabType, string>
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void
  setActiveTab: (tab: TabType) => void
  setLoading: (loading: boolean) => void
  setCustomerId: (id: string) => void
  setApiKey: (key: string) => void
  clearMessages: () => void
  setResearchStages: (stages: ResearchStage[]) => void
  updateResearchStage: (key: string, update: Partial<ResearchStage>) => void
  setStreaming: (v: boolean) => void
  getContextMessages: (limit?: number) => Message[]
  newThread: () => void
  /** 切换主题 */
  toggleTheme: () => void
  /** 移除最后一条AI消息（用于重新生成） */
  removeLastAssistantMessage: () => string | null
  /** 设置当前登录用户 */
  setAuthUser: (user: AuthUser | null) => void
  /** Toast通知管理 */
  addToast: (toast: Omit<ToastItem, 'id'>) => void
  removeToast: (id: string) => void
  /** 消息反馈 */
  setMessageFeedback: (msgId: string, feedback: 1 | -1 | 0) => void
}

function persist(state: Partial<ChatStore>) {
  try {
    localStorage.setItem('wealth-advisor-state', JSON.stringify({
      messages: state.messages,
      activeTab: state.activeTab,
      customerId: state.customerId,
      apiKey: state.apiKey,
      threadId: state.threadId,
      theme: state.theme,
      authUser: state.authUser,  // 修复：持久化登录用户信息
    }))
  } catch { /* 忽略存储错误 */ }
}

export const useChatStore = create<ChatStore>((set, get) => {
  const persisted = loadPersistedState()

  return {
    messages: persisted.messages || [],
    activeTab: (persisted.activeTab as TabType) || 'fund-qa',
    isLoading: false,
    customerId: persisted.customerId || 'customer1',
    apiKey: persisted.apiKey || '',
    researchStages: [],
    isStreaming: false,
    threadId: persisted.threadId || crypto.randomUUID(),
    theme: (persisted.theme as ThemeType) || 'light',
    authUser: (persisted.authUser as AuthUser) || null,
    toasts: [],
    tabMessages: { 'fund-qa': persisted.messages || [], 'research': [], 'wealth-advisor': [] },
    tabThreadIds: { 'fund-qa': persisted.threadId || crypto.randomUUID(), 'research': crypto.randomUUID(), 'wealth-advisor': crypto.randomUUID() },

    addMessage: (message) =>
      set((state) => {
        const next = {
          messages: [...state.messages, { ...message, id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`, timestamp: Date.now() }],
        }
        persist({ ...state, ...next })
        return next
      }),

    setActiveTab: (tab) => {
      const state = get()
      // 保存当前Tab的消息和线程ID
      const newTabMessages = { ...state.tabMessages, [state.activeTab]: state.messages }
      const newTabThreadIds = { ...state.tabThreadIds, [state.activeTab]: state.threadId }
      // 切换到目标Tab，恢复其消息
      const next = {
        activeTab: tab,
        messages: newTabMessages[tab] || [],
        researchStages: [] as ResearchStage[],
        threadId: newTabThreadIds[tab] || crypto.randomUUID(),
        tabMessages: newTabMessages,
        tabThreadIds: newTabThreadIds,
      }
      set(next)
      persist({ ...state, ...next })
    },

    setLoading: (loading) => set({ isLoading: loading }),
    setCustomerId: (id) => { set({ customerId: id }); persist(get()) },
    setApiKey: (key) => { set({ apiKey: key }); persist(get()) },
    clearMessages: () => set({ messages: [] }),

    setResearchStages: (stages) => set({ researchStages: stages }),
    updateResearchStage: (key, update) =>
      set((state) => ({
        researchStages: state.researchStages.map((s) => (s.key === key ? { ...s, ...update } : s)),
      })),
    setStreaming: (v) => set({ isStreaming: v }),

    getContextMessages: (limit = 6) => {
      const all = get().messages
      return all.slice(-limit)
    },

    newThread: () => {
      const next = { threadId: crypto.randomUUID(), messages: [] as Message[], researchStages: [] as ResearchStage[] }
      set(next)
      persist({ ...get(), ...next })
    },

    toggleTheme: () => {
      const next = { theme: get().theme === 'light' ? 'dark' as ThemeType : 'light' as ThemeType }
      set(next)
      persist(get())
      document.documentElement.setAttribute('data-theme', next.theme)
    },

    removeLastAssistantMessage: () => {
      const msgs = get().messages
      // 找到最后一条用户消息并返回
      const lastUserIdx = [...msgs].reverse().findIndex((m) => m.role === 'user')
      if (lastUserIdx === -1) return null
      const lastUserMsg = msgs[msgs.length - 1 - lastUserIdx]
      // 移除最后一条用户消息之后的所有消息
      const next = { messages: msgs.slice(0, msgs.length - 1 - lastUserIdx) }
      set(next)
      persist({ ...get(), ...next })
      return lastUserMsg.content
    },

    addToast: (toast) =>
      set((state) => ({
        toasts: [...state.toasts, { ...toast, id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}` }],
      })),

    removeToast: (id) =>
      set((state) => ({
        toasts: state.toasts.filter((t) => t.id !== id),
      })),

    setMessageFeedback: (msgId, feedback) =>
      set((state) => {
        const next = {
          messages: state.messages.map((m) =>
            m.id === msgId ? { ...m, feedback } : m
          ),
        }
        persist({ ...state, ...next })
        return next
      }),

    setAuthUser: (user) =>
      set((state) => {
        const next: Partial<ChatStore> = { authUser: user }
        if (user === null) {
          next.messages = []
        }
        persist({ ...state, ...next })
        return next
      }),
  }
})

// 初始化主题
if (typeof document !== 'undefined') {
  const saved = loadPersistedState()
  document.documentElement.setAttribute('data-theme', saved.theme || 'light')
}