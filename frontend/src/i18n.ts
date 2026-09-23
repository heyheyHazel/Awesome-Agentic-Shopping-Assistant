import { createContext, useContext } from 'react'

export type Lang = 'en' | 'zh'

export const STRINGS = {
  en: {
    // header
    lang_en: 'EN',
    lang_zh: '中文',
    // agent panel
    agents_label: 'AI Agents',
    agent_supervisor: 'Supervisor Agent',
    agent_supervisor_desc: 'Plans, routes tasks and coordinates all agents.',
    agent_recommendation: 'Recommendation Agent',
    agent_recommendation_desc: 'Recalls candidates and re-ranks them for you.',
    agent_copywriting: 'Copywriting Agent',
    agent_copywriting_desc: 'Generates product copy and messages.',
    agent_inventory: 'Inventory Agent',
    agent_inventory_desc: 'Checks stock, price and availability in real-time.',
    agent_assistant: 'Assistant',
    // chat panel
    chat_title: 'Live Agent Conversation',
    sse_active: 'SSE Streaming…',
    sse_idle: 'SSE Streaming Connected',
    new_chat: 'New chat',
    empty_title: 'Ask me anything about products',
    empty_desc: 'Five specialised agents will plan, recall, rank, check stock and write copy for you.',
    suggest_1: 'Recommend a skincare routine under ¥400.',
    suggest_2: 'A gift for a friend who loves coffee.',
    suggest_3: 'What home goods would you recommend?',
    input_placeholder: 'Ask me anything about products...',
    send: 'Send',
    // product cards
    badge_best: 'Best Match',
    badge_rated: 'High Rated',
    badge_value: 'Great Value',
    // inventory
    stock_in: 'In Stock ({n})',
    stock_low: 'Low Stock ({n})',
    stock_out: 'Sold Out',
    inv_all_in_stock: 'All items are in stock and ready to ship.',
    inv_low: 'Low stock on {names} — they may sell out soon.',
    inv_out: 'Not available right now: {names}.',
    // profile panel
    profile_title: 'User Profile',
    show_full: 'Show full profile',
    hide_full: 'Hide full profile',
    row_segment: 'RFM Segment',
    row_recency: 'Recency',
    row_frequency: 'Frequency',
    row_monetary: 'Monetary',
    row_rfm_score: 'RFM Score',
    row_preferred: 'Preferred',
    row_budget: 'Budget',
    recency_value: '{n} days ago',
    frequency_value: '{n} orders',
    rfm_clustering: 'RFM Clustering',
    current: 'current',
    ab_title: 'A/B Testing · Thompson Sampling',
    ab_conversion: 'Conversion Rate',
    ab_bucket: 'your bucket',
    ab_winner: 'Winner: Variant {name}',
    response_time: 'Response Time',
    waiting_backend: 'Waiting for the backend…',
    // footer
    feature_collab_title: 'Multi-Agent Collaboration',
    feature_collab_desc: 'Supervisor plans, routes tasks and coordinates specialised agents.',
    feature_personal_title: 'Personalized Experience',
    feature_personal_desc: 'RFM clustering and user profiling drive every ranking.',
    feature_decision_title: 'Smarter Decisions',
    feature_decision_desc: 'Thompson Sampling powers data-driven A/B testing.',
    feature_stream_title: 'Real-time Streaming',
    feature_stream_desc: 'SSE streams every agent step for instant feedback.',
  },
  zh: {
    // header
    lang_en: 'EN',
    lang_zh: '中文',
    // agent panel
    agents_label: 'AI Agents',
    agent_supervisor: '调度 Agent',
    agent_supervisor_desc: '负责任务规划、动态路由，协调所有 Agent。',
    agent_recommendation: '推荐 Agent',
    agent_recommendation_desc: '召回候选商品，并按你的偏好精排。',
    agent_copywriting: '文案 Agent',
    agent_copywriting_desc: '生成个性化商品文案与推荐话术。',
    agent_inventory: '库存 Agent',
    agent_inventory_desc: '实时校验库存、价格与可售状态。',
    agent_assistant: '购物助手',
    // chat panel
    chat_title: '实时 Agent 对话',
    sse_active: 'SSE 传输中…',
    sse_idle: 'SSE 已连接',
    new_chat: '新对话',
    empty_title: '问我任何商品问题',
    empty_desc: '五个专业 Agent 会为你完成规划、召回、精排、库存校验与文案撰写。',
    suggest_1: '推荐一套 400 元以内的护肤品。',
    suggest_2: '送朋友的礼物，对方喜欢咖啡。',
    suggest_3: '有什么家居好物推荐吗？',
    input_placeholder: '输入你想买的东西…',
    send: '发送',
    // product cards
    badge_best: '最佳匹配',
    badge_rated: '高分好评',
    badge_value: '超值之选',
    // inventory
    stock_in: '有货（{n}）',
    stock_low: '库存紧张（{n}）',
    stock_out: '已售罄',
    inv_all_in_stock: '所有商品均有现货，可立即发货。',
    inv_low: '{names} 库存紧张，可能很快售罄。',
    inv_out: '暂时缺货：{names}。',
    // profile panel
    profile_title: '用户画像',
    show_full: '展开完整画像',
    hide_full: '收起完整画像',
    row_segment: 'RFM 客群',
    row_recency: '最近购买',
    row_frequency: '购买频次',
    row_monetary: '消费金额',
    row_rfm_score: 'RFM 得分',
    row_preferred: '偏好类目',
    row_budget: '预算区间',
    recency_value: '{n} 天前',
    frequency_value: '{n} 单',
    rfm_clustering: 'RFM 客群聚类',
    current: '当前',
    ab_title: 'A/B 测试 · Thompson Sampling',
    ab_conversion: '转化率',
    ab_bucket: '你的分组',
    ab_winner: '胜出：变体 {name}',
    response_time: '响应耗时',
    waiting_backend: '等待后端连接…',
    // footer
    feature_collab_title: '多 Agent 协作',
    feature_collab_desc: '调度 Agent 负责任务规划、路由与协调。',
    feature_personal_title: '个性化体验',
    feature_personal_desc: 'RFM 客群聚类与用户画像驱动每一次排序。',
    feature_decision_title: '智能决策',
    feature_decision_desc: 'Thompson Sampling 驱动数据化 A/B 测试。',
    feature_stream_title: '实时流式',
    feature_stream_desc: 'SSE 实时推送每个 Agent 的执行过程。',
  },
} as const

export type StringKey = keyof (typeof STRINGS)['en']

export const CATEGORY_LABELS: Record<string, string> = {
  'Running Shoes': '跑鞋',
  Sneakers: '休闲鞋',
  Audio: '音频',
  Wearables: '智能穿戴',
  Beauty: '美妆',
  Skincare: '护肤',
  Fashion: '服饰',
  Home: '家居',
  Kitchen: '厨具',
  Sports: '运动',
  Outdoor: '户外',
  Accessories: '配饰',
  Gaming: '游戏',
  Food: '食品',
}

/**
 * The downloaded ShopSimulator catalog stores categories in Chinese while the
 * built-in demo catalog uses the English keys above, so the English UI needs
 * the reverse direction too.
 */
export const CATEGORY_EN: Record<string, string> = {
  家居家装: 'Home & Living',
  美妆个护健康: 'Beauty & Care',
  生产材料农用品: 'Supplies',
  休闲娱乐文教: 'Leisure',
  服饰鞋包饰品: 'Clothing & Accessories',
  家用电器数码: 'Electronics',
  食品饮品: 'Food & Drink',
  运动户外交通: 'Sports & Outdoors',
  母婴儿童: 'Kids & Baby',
}

export const TAG_LABELS: Record<string, string> = {
  lightweight: '轻量',
  breathable: '透气',
  'daily training': '日常训练',
  cushioning: '缓震',
  durable: '耐用',
  comfortable: '舒适',
  stable: '稳定',
  trail: '越野',
  waterproof: '防水',
  grip: '抓地',
  budget: '平价',
  classic: '经典',
  casual: '休闲',
  streetwear: '街头',
  retro: '复古',
  'noise cancelling': '降噪',
  wireless: '无线',
  earbuds: '入耳式',
  'over-ear': '头戴式',
  'fitness tracker': '运动手环',
  'heart rate': '心率',
  gps: 'GPS',
  'sports watch': '运动手表',
  lipstick: '口红',
  'long lasting': '持久',
  serum: '精华',
  brightening: '提亮',
  foundation: '粉底',
  'natural finish': '自然妆感',
  toner: '爽肤水',
  hydrating: '保湿',
  'anti-aging': '抗老',
  'night cream': '晚霜',
  cotton: '纯棉',
  basic: '基础款',
  dress: '连衣裙',
  summer: '夏季',
  coat: '大衣',
  aroma: '香薰',
  'home decor': '家居装饰',
  sleep: '助眠',
  calming: '舒缓',
  kitchen: '厨房',
  ceramic: '陶瓷',
  coffee: '咖啡',
  'manual brew': '手冲',
  yoga: '瑜伽',
  'non-slip': '防滑',
  strength: '力量',
  adjustable: '可调节',
  hiking: '徒步',
  camping: '露营',
  portable: '便携',
  leather: '皮革',
  slim: '轻薄',
  'uv protection': '防紫外线',
  polarized: '偏光',
  ergonomic: '人体工学',
  arabica: '阿拉比卡',
}

export const SEGMENT_LABELS: Record<string, string> = {
  Champions: '冠军客群',
  Loyal: '忠诚客群',
  Potential: '潜力客群',
  'At Risk': '流失风险',
  New: '新客',
}

export function fillTemplate(template: string, vars: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (_, key: string) => String(vars[key] ?? ''))
}

export function initialLang(): Lang {
  const fromUrl = new URLSearchParams(window.location.search).get('lang')
  if (fromUrl === 'zh' || fromUrl === 'en') return fromUrl
  const stored = window.localStorage.getItem('lang')
  if (stored === 'zh' || stored === 'en') return stored
  return 'en'
}

export interface I18n {
  lang: Lang
  setLang: (lang: Lang) => void
  /** Look up a UI string. */
  t: (key: StringKey) => string
  /** Look up a UI string containing {placeholders}. */
  fill: (key: StringKey, vars: Record<string, string | number>) => string
  /** Translate a catalog category. */
  category: (value: string) => string
  /** Translate a product tag. */
  tag: (value: string) => string
  /** Translate an RFM segment name. */
  segment: (value: string) => string
}

export const I18nContext = createContext<I18n | null>(null)


/** Build the i18n object for one language (used by the provider). */
export function buildI18n(lang: Lang, setLang: (lang: Lang) => void): I18n {
  const pick = (key: StringKey) => STRINGS[lang][key] ?? STRINGS.en[key] ?? key
  return {
    lang,
    setLang,
    t: (key) => pick(key),
    fill: (key, vars) => fillTemplate(pick(key), vars),
    category: (value) =>
      lang === 'zh' ? (CATEGORY_LABELS[value] ?? value) : (CATEGORY_EN[value] ?? value),
    tag: (value) => (lang === 'zh' ? (TAG_LABELS[value] ?? value) : value),
    segment: (value) => (lang === 'zh' ? (SEGMENT_LABELS[value] ?? value) : value),
  }
}

export function useI18n(): I18n {
  const context = useContext(I18nContext)
  if (!context) throw new Error('useI18n must be used inside LanguageProvider')
  return context
}
