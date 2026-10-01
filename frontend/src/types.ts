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

export interface ToolEvent {
  tool: string
  status: 'running' | 'done'
  message: string
}

/** SSE payload for the products the agent chose to present. */
export interface ProductsEvent {
  products: Product[]
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
  // Stock is attached to the cards rather than shown as its own message: the agent
  // checks stock before it presents, so a standalone bubble would appear before any
  // product is on screen and read as if it came out of nowhere.
  | { kind: 'products'; id: string; products: Product[]; time: string; stock?: InventoryItem[] }
