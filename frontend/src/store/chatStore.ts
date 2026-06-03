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

  setActiveTab: (tab: TabKey) => void
  addMessage: (msg: Message) => void
  setLoading: (loading: boolean) => void
  setCustomerId: (id: string) => void
  clearMessages: () => void
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  activeTab: 'fund-qa',
  isLoading: false,
  customerId: 'customer1',

  setActiveTab: (tab) => set({ activeTab: tab, messages: [] }),
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  setLoading: (loading) => set({ isLoading: loading }),
  setCustomerId: (id) => set({ customerId: id }),
  clearMessages: () => set({ messages: [] }),
}))