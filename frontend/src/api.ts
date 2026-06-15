const API_BASE = '/api/v1'
const DEFAULT_TIMEOUT = 120000 // 默认请求超时 120秒

// ========== 认证管理 ==========

const TOKEN_KEY = 'wealth_advisor_tokens'

interface TokenStore {
  accessToken: string
  refreshToken: string
  expiresAt: number // 过期时间戳
}

let _tokenStore: TokenStore | null = null

function getTokenStore(): TokenStore | null {
  if (_tokenStore) return _tokenStore
  try {
    const raw = localStorage.getItem(TOKEN_KEY)
    if (raw) {
      _tokenStore = JSON.parse(raw)
      return _tokenStore
    }
  } catch { /* ignore */ }
  return null
}

function setTokenStore(store: TokenStore) {
  _tokenStore = store
  localStorage.setItem(TOKEN_KEY, JSON.stringify(store))
}

/**
 * 带超时控制的 fetch 封装
 * 解决浏览器默认超时过长（通常120s+）导致用户体验差的问题
 */
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeout: number = DEFAULT_TIMEOUT): Promise<Response> {
  const controller = new AbortController()
  // 合并外部 signal 和内部 timeout signal
  const existingSignal = options.signal
  if (existingSignal) {
    existingSignal.addEventListener('abort', () => controller.abort(existingSignal.reason))
  }
  const timeoutId = setTimeout(() => controller.abort(new Error('请求超时')), timeout)
  try {
    const response = await fetch(url, { ...options, signal: controller.signal })
    return response
  } finally {
    clearTimeout(timeoutId)
  }
}

function clearTokenStore() {
  _tokenStore = null
  localStorage.removeItem(TOKEN_KEY)
}

export function getAccessToken(): string | null {
  const store = getTokenStore()
  if (!store || Date.now() > store.expiresAt) {
    return null
  }
  return store.accessToken
}

function getHeaders(apiKey?: string): HeadersInit {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }
  const token = getAccessToken()
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  } else if (apiKey) {
    headers['X-API-Key'] = apiKey
  }
  return headers
}

// ========== 认证API ==========

export interface AuthUser {
  id: number
  username: string
  display_name: string
}

export interface AuthResponse {
  access_token: string
  refresh_token: string
  token_type: string
  user: AuthUser
}

export async function register(
  username: string,
  password: string,
  displayName: string = '',
): Promise<AuthResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password, display_name: displayName }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: '注册失败' }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  const data: AuthResponse = await res.json()
  // 存储token，设定过期时间（2小时减去5分钟缓冲）
  setTokenStore({
    accessToken: data.access_token,
    refreshToken: data.refresh_token,
    expiresAt: Date.now() + (2 * 60 - 5) * 60 * 1000,
  })
  return data
}

export async function login(username: string, password: string): Promise<AuthResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: '登录失败' }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  const data: AuthResponse = await res.json()
  setTokenStore({
    accessToken: data.access_token,
    refreshToken: data.refresh_token,
    expiresAt: Date.now() + (2 * 60 - 5) * 60 * 1000,
  })
  return data
}

export async function refreshAccessToken(): Promise<AuthResponse | null> {
  const store = getTokenStore()
  if (!store?.refreshToken) return null
  try {
    const res = await fetchWithTimeout(`${API_BASE}/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: store.refreshToken }),
    })
    if (!res.ok) {
      clearTokenStore()
      return null
    }
    const data: AuthResponse = await res.json()
    setTokenStore({
      accessToken: data.access_token,
      refreshToken: data.refresh_token,
      expiresAt: Date.now() + (2 * 60 - 5) * 60 * 1000,
    })
    return data
  } catch {
    return null
  }
}

export async function getCurrentUser(): Promise<AuthUser | null> {
  const token = getAccessToken()
  if (!token) return null
  try {
    const res = await fetchWithTimeout(`${API_BASE}/auth/me`, {
      headers: { 'Authorization': `Bearer ${token}` },
    })
    if (!res.ok) return null
    return res.json()
  } catch {
    return null
  }
}

export function logout() {
  clearTokenStore()
}

// ========== 业务API ==========

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

export async function fundQA(query: string, threadId?: string, apiKey?: string, signal?: AbortSignal): Promise<FundQAResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/fund-qa`, {
    method: 'POST',
    headers: getHeaders(apiKey),
    body: JSON.stringify({ query, thread_id: threadId || null }),
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
  threadId?: string,
  apiKey?: string,
  signal?: AbortSignal,
): Promise<ResearchResponse> {
  const res = await fetchWithTimeout(`${API_BASE}/research`, {
    method: 'POST',
    headers: getHeaders(apiKey),
    body: JSON.stringify({ topic, industry, horizon, thread_id: threadId || null }),
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
  const res = await fetchWithTimeout(`${API_BASE}/wealth-advisor`, {
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

/** 流式投研分析 - 通过SSE实时获取五阶段进度，支持多轮对话 */
export async function researchStream(
  topic: string,
  industry: string = '综合',
  horizon: string = '中期',
  threadId?: string,
  apiKey?: string,
  onStage?: (stage: string, name: string, message: string) => void,
  onStageDone?: (stage: string, name: string, summary: string) => void,
  onComplete?: (report: string) => void,
  onError?: (error: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const params = new URLSearchParams({ topic, industry, horizon })
  if (threadId) {
    params.set('thread_id', threadId)
  }
  const headers: HeadersInit = {}
  if (apiKey) {
    headers['X-API-Key'] = apiKey
  }
  const url = `${API_BASE}/research/stream?${params}`

  const response = await fetch(url, { headers, signal })
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }

  const reader = response.body?.getReader()
  if (!reader) throw new Error('无法读取流数据')

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    if (signal?.aborted) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      try {
        const event = JSON.parse(line.slice(6))
        if (event.event === 'stage_start') {
          onStage?.(event.stage, event.name, event.message)
        } else if (event.event === 'stage_done') {
          onStageDone?.(event.stage, event.name, event.summary)
        } else if (event.event === 'complete') {
          onComplete?.(event.report)
        } else if (event.error) {
          onError?.(event.error)
        }
      } catch {
        // 忽略解析错误
      }
    }
  }
}

/** 加载历史对话线程的消息列表 */
export async function loadThreadMessages(threadId: string): Promise<Array<{ role: string; content: string }>> {
  const res = await fetchWithTimeout(`/api/v1/conversations/${threadId}`)
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`)
  }
  return res.json()
}