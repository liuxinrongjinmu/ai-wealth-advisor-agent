import { create } from 'zustand'

export type TabType = 'fund-qa' | 'research' | 'wealth-advisor'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
}

interface ChatStore {
  messages: Message[]
  activeTab: TabType
  isLoading: boolean
  customerId: string
  apiKey: string
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void
  setActiveTab: (tab: TabType) => void
  setLoading: (loading: boolean) => void
  setCustomerId: (id: string) => void
  setApiKey: (key: string) => void
  clearMessages: () => void
}

export const useChatStore = create<ChatStore>((set) => ({
  messages: [],
  activeTab: 'fund-qa',
  isLoading: false,
  customerId: 'customer1',
  apiKey: '',
  addMessage: (message) =>
    set((state) => ({
      messages: [
        ...state.messages,
        {
          ...message,
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
          timestamp: Date.now(),
        },
      ],
    })),
  setActiveTab: (tab) => set({ activeTab: tab, messages: [] }),
  setLoading: (loading) => set({ isLoading: loading }),
  setCustomerId: (id) => set({ customerId: id }),
  setApiKey: (key) => set({ apiKey: key }),
  clearMessages: () => set({ messages: [] }),
}))
