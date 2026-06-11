const API_BASE = '/api/v1'

function getHeaders(apiKey?: string): HeadersInit {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }
  if (apiKey) {
    headers['X-API-Key'] = apiKey
  }
  return headers
}

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

export async function fundQA(query: string, apiKey?: string, signal?: AbortSignal): Promise<FundQAResponse> {
  const res = await fetch(`${API_BASE}/fund-qa`, {
    method: 'POST',
    headers: getHeaders(apiKey),
    body: JSON.stringify({ query }),
    signal,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: '请求失败' }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function research(
  topic: string,
  industry: string = '综合',
  horizon: string = '中期',
  apiKey?: string,
  signal?: AbortSignal,
): Promise<ResearchResponse> {
  const res = await fetch(`${API_BASE}/research`, {
    method: 'POST',
    headers: getHeaders(apiKey),
    body: JSON.stringify({ topic, industry, horizon }),
    signal,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: '请求失败' }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function wealthAdvisor(
  query: string,
  customerId: string = 'customer1',
  threadId?: string,
  apiKey?: string,
  signal?: AbortSignal,
): Promise<WealthAdvisorResponse> {
  const res = await fetch(`${API_BASE}/wealth-advisor`, {
    method: 'POST',
    headers: getHeaders(apiKey),
    body: JSON.stringify({ query, customer_id: customerId, thread_id: threadId }),
    signal,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: '请求失败' }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}