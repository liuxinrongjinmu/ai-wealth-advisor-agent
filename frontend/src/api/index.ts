const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8002'

export interface FundQAResponse {
  query: string
  answer: string
  category: string
}

export interface ResearchResponse {
  topic: string
  industry: string
  horizon: string
  report: string
  error?: string
}

export interface WealthAdvisorResponse {
  query: string
  response: string
  processing_mode: string
  error?: string
}

async function post<T>(path: string, body: Record<string, string>): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`请求失败 (${res.status}): ${text}`)
  }
  return res.json()
}

export async function askFundQA(query: string): Promise<FundQAResponse> {
  return post<FundQAResponse>('/api/fund-qa', { query })
}

export async function askResearch(
  topic: string,
  industry: string = '综合',
  horizon: string = '中期'
): Promise<ResearchResponse> {
  return post<ResearchResponse>('/api/research', {
    topic,
    industry,
    horizon,
  })
}

export async function askWealthAdvisor(
  query: string,
  customerId: string,
  threadId?: string
): Promise<WealthAdvisorResponse> {
  const body: Record<string, string> = {
    query,
    customer_id: customerId,
  }
  if (threadId) {
    body.thread_id = threadId
  }
  return post<WealthAdvisorResponse>('/api/wealth-advisor', body)
}