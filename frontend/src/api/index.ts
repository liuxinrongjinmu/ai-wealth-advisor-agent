const API_BASE = 'http://localhost:8002'

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

export async function askResearch(topic: string): Promise<ResearchResponse> {
  return post<ResearchResponse>('/api/research', {
    topic,
    industry: '综合',
    horizon: '中期',
  })
}

export async function askWealthAdvisor(
  query: string,
  customerId: string
): Promise<WealthAdvisorResponse> {
  return post<WealthAdvisorResponse>('/api/wealth-advisor', {
    query,
    customer_id: customerId,
  })
}