export interface Product {
  product_id: string
  name: string
  category: string
  price: number
  brand: string
  stock: number
  rating: number
  rating_count: number
  tags: string[]
}

export interface RFM {
  recency_days: number
  orders: number
  lifetime_value: number
  recency: number
  frequency: number
  monetary: number
  overall: number
}

export interface UserProfile {
  user_id: string
  name: string
  email: string
  tier: string
  segment: string
  rfm: RFM
  preferred_categories: string[]
  price_range: [number, number]
}

export interface ProfileResponse {
  profile: UserProfile
  segments: string[]
}

export interface InventoryItem {
  product_id: string
  name: string
  stock: number
  status: 'in_stock' | 'low_stock' | 'out_of_stock'
  purchase_limit: number | null
}

export interface CopyItem {
  product_id: string
  text: string
}

export interface VariantStats {
  name: string
  label: string
  trials: number
  conversion_rate: number
}

export interface ExperimentInfo {
  experiment_id: string
  name: string
  variant: string | null
  variants: VariantStats[]
  winner: string
}

export interface UserSummary {
  user_id: string
  name: string
  email: string
  tier: string
}

export interface MetaInfo {
  data_source: string
  currency: string
  currency_symbol: string
}

// ── SSE payloads ──────────────────────────────────────────────────────

export interface AgentEvent {
  agent: string
  status: 'running' | 'done'
  message: string
}

export interface PlanEvent {
  intent: 'product_search' | 'general'
  reply: string
  agents: string[]
  variant: string
}

export interface MarketingEvent {
  items: CopyItem[]
  segment: string
}

export interface InventoryEvent {
  items: InventoryItem[]
  summary: string
}

export interface DoneEvent {
  latency_ms: number
  timings: Record<string, number>
}

// ── chat feed ─────────────────────────────────────────────────────────

export type AgentStatus = 'idle' | 'running' | 'done'

export type FeedItem =
  | { kind: 'user'; id: string; text: string; time: string }
  | { kind: 'agent'; id: string; agent: string; text: string; time: string; streaming?: boolean }
  | { kind: 'products'; id: string; products: Product[]; time: string }
  | { kind: 'inventory'; id: string; items: InventoryItem[]; summary: string; time: string }
