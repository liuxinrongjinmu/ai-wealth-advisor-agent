import { create } from 'zustand'

export type TabKey = 'fund-qa' | 'research' | 'wealth-advisor'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
}

interface ChatState {
  messages: Message[]
  activeTab: TabKey
  isLoading: boolean
  customerId: string
  threadId: string

  setActiveTab: (tab: TabKey) => void
  addMessage: (msg: Message) => void
  setLoading: (loading: boolean) => void
  setCustomerId: (id: string) => void
  setThreadId: (id: string) => void
  clearMessages: () => void
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  activeTab: 'fund-qa',
  isLoading: false,
  customerId: 'customer1',
  threadId: `thread-${Date.now()}`,

  setActiveTab: (tab) => set({ activeTab: tab, messages: [], threadId: `thread-${Date.now()}` }),
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  setLoading: (loading) => set({ isLoading: loading }),
  setCustomerId: (id) => set({ customerId: id }),
  setThreadId: (id) => set({ threadId: id }),
  clearMessages: () => set({ messages: [], threadId: `thread-${Date.now()}` }),
}))