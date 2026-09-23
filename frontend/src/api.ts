import type { ExperimentInfo, ProfileResponse, UserSummary } from './types'

const JSON_HEADERS = { 'Content-Type': 'application/json' }

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${url} failed: ${response.status}`)
  return response.json() as Promise<T>
}

export function fetchUsers(): Promise<UserSummary[]> {
  return getJson<UserSummary[]>('/api/v1/users')
}

export function fetchProfile(userId: string): Promise<ProfileResponse> {
  return getJson<ProfileResponse>(`/api/v1/users/${userId}/profile`)
}

export function fetchExperiment(userId: string): Promise<ExperimentInfo> {
  return getJson<ExperimentInfo>(`/api/v1/experiments?user_id=${userId}`)
}

export interface ChatStreamParams {
  userId: string
  message: string
  threadId: string | null
  signal: AbortSignal
}

/** POST /api/v1/chat and dispatch each SSE event to the listener. */
export async function streamChat(
  params: ChatStreamParams,
  onEvent: (event: string, data: unknown) => void,
): Promise<void> {
  const response = await fetch('/api/v1/chat', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({
      user_id: params.userId,
      message: params.message,
      thread_id: params.threadId,
    }),
    signal: params.signal,
  })
  if (!response.ok || !response.body) {
    throw new Error(`chat failed: ${response.status}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let eventName = 'message'

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        eventName = line.slice(7).trim()
      } else if (line.startsWith('data: ')) {
        try {
          onEvent(eventName, JSON.parse(line.slice(6)))
        } catch {
          // ignore malformed frames
        }
      }
    }
  }
}
