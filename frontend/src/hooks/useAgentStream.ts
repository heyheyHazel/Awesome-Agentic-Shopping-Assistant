import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchExperiment, fetchProfile, streamChat } from '../api'
import type { Lang } from '../i18n'
import type {
  AgentEvent,
  AgentStatus,
  DoneEvent,
  ExperimentInfo,
  FeedItem,
  InventoryEvent,
  PlanEvent,
  ProfileResponse,
  Product,
} from '../types'

function now(): string {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

let uid = 0
function nextId(): string {
  uid += 1
  return `item-${uid}`
}

/**
 * Owns the SSE connection to `/api/v1/chat` and keeps every panel in sync:
 * chat feed, per-agent status, user profile, A/B experiment and latency.
 */
export function useAgentStream(userId: string, language: Lang) {
  const [feed, setFeed] = useState<FeedItem[]>([])
  const [agentStates, setAgentStates] = useState<Record<string, AgentStatus>>({})
  const [profile, setProfile] = useState<ProfileResponse | null>(null)
  const [experiment, setExperiment] = useState<ExperimentInfo | null>(null)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const [timings, setTimings] = useState<Record<string, number> | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)

  const threadRef = useRef<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Load the right panel data whenever the shopper changes (stale responses are ignored).
  useEffect(() => {
    let cancelled = false
    Promise.all([fetchProfile(userId), fetchExperiment(userId)])
      .then(([profileData, experimentData]) => {
        if (cancelled) return
        setProfile(profileData)
        setExperiment(experimentData)
      })
      .catch(() => {
        /* backend offline — panels stay empty */
      })
    return () => {
      cancelled = true
    }
  }, [userId])

  const send = useCallback(
    async (text: string) => {
      if (!text.trim() || isStreaming) return

      setFeed((prev) => [...prev, { kind: 'user', id: nextId(), text: text.trim(), time: now() }])
      setAgentStates({})
      setLatencyMs(null)
      setTimings(null)
      setIsStreaming(true)

      const controller = new AbortController()
      abortRef.current = controller
      let assistantId: string | null = null

      try {
        await streamChat(
          { userId, message: text.trim(), threadId: threadRef.current, language, signal: controller.signal },
          (event, data) => {
            switch (event) {
              case 'session':
                threadRef.current = (data as { thread_id: string }).thread_id
                break
              case 'agent': {
                const payload = data as AgentEvent
                setAgentStates((prev) => ({ ...prev, [payload.agent]: payload.status }))
                break
              }
              case 'plan': {
                const payload = data as PlanEvent
                if (payload.intent === 'product_search') {
                  setFeed((prev) => [
                    ...prev,
                    { kind: 'agent', id: nextId(), agent: 'supervisor', text: payload.reply, time: now() },
                  ])
                }
                break
              }
              case 'profile': {
                const payload = data as { profile: ProfileResponse['profile'] }
                setProfile((prev) => ({
                  profile: payload.profile,
                  segments: prev?.segments ?? [],
                }))
                break
              }
              case 'products': {
                const payload = data as { products: Product[] }
                setFeed((prev) => [
                  ...prev,
                  { kind: 'products', id: nextId(), products: payload.products, time: now() },
                ])
                break
              }
              case 'inventory': {
                const payload = data as InventoryEvent
                setFeed((prev) => [
                  ...prev,
                  { kind: 'inventory', id: nextId(), items: payload.items, summary: payload.summary, time: now() },
                ])
                break
              }
              case 'experiment':
                setExperiment(data as ExperimentInfo)
                break
              case 'token': {
                const payload = data as { content: string }
                if (!assistantId) {
                  assistantId = nextId()
                  setFeed((prev) => [
                    ...prev,
                    { kind: 'agent', id: assistantId as string, agent: 'assistant', text: payload.content, time: now(), streaming: true },
                  ])
                } else {
                  setFeed((prev) =>
                    prev.map((item) =>
                      item.id === assistantId && item.kind === 'agent'
                        ? { ...item, text: item.text + payload.content }
                        : item,
                    ),
                  )
                }
                break
              }
              case 'done': {
                const payload = data as DoneEvent
                setLatencyMs(payload.latency_ms)
                setTimings(payload.timings)
                break
              }
              case 'error': {
                const payload = data as { message: string }
                setFeed((prev) => [
                  ...prev,
                  { kind: 'agent', id: nextId(), agent: 'assistant', text: payload.message, time: now() },
                ])
                break
              }
            }
          },
        )
      } catch (error) {
        if (!(error instanceof DOMException && error.name === 'AbortError')) {
          setFeed((prev) => [
            ...prev,
            { kind: 'agent', id: nextId(), agent: 'assistant', text: 'Connection lost — please try again.', time: now() },
          ])
        }
      } finally {
        setFeed((prev) =>
          prev.map((item) => (item.kind === 'agent' ? { ...item, streaming: false } : item)),
        )
        setIsStreaming(false)
        abortRef.current = null
      }
    },
    [isStreaming, userId, language],
  )

  const newChat = useCallback(() => {
    abortRef.current?.abort()
    threadRef.current = null
    setFeed([])
    setAgentStates({})
    setLatencyMs(null)
    setTimings(null)
  }, [])

  return { feed, agentStates, profile, experiment, latencyMs, timings, isStreaming, send, newChat }
}
