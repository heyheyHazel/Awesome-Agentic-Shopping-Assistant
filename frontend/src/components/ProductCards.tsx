import { Star } from 'lucide-react'
import { useI18n } from '../i18n'
import type { I18n, StringKey } from '../i18n'
import type { InventoryItem, Product } from '../types'

const CATEGORY_EMOJI: Record<string, string> = {
  'Running Shoes': '👟',
  Sneakers: '👟',
  Audio: '🎧',
  Wearables: '⌚',
  Beauty: '💄',
  Skincare: '🧴',
  Fashion: '👗',
  Home: '🕯️',
  Kitchen: '🍳',
  Sports: '🧘',
  Outdoor: '🎒',
  Accessories: '🕶️',
  Gaming: '🎮',
  Food: '☕',
  // ShopSimulator (Chinese catalog) domains
  家居家装: '🕯️',
  美妆个护健康: '💄',
  生产材料农用品: '🧰',
  休闲娱乐文教: '🎨',
  服饰鞋包饰品: '👗',
  家用电器数码: '🔌',
  食品饮品: '☕',
  运动户外交通: '🎒',
  母婴儿童: '🧸',
}

const BADGE_TONES: Record<string, string> = {
  brand: 'bg-brand-soft text-brand',
  success: 'bg-success-soft text-success',
  warning: 'bg-warning-soft text-warning',
}

const STOCK_TONES: Record<InventoryItem['status'], string> = {
  in_stock: 'border-success/25 bg-success-soft text-success',
  low_stock: 'border-warning/25 bg-warning-soft text-warning',
  out_of_stock: 'border-danger/25 bg-red-50 text-danger',
}

/** First result = Best Match, next best rating = High Rated, cheapest remaining = Great Value. */
function computeBadges(products: Product[]) {
  const badges = new Map<string, { label: StringKey; tone: string }>()
  const assigned = new Set<string>()
  if (products.length === 0) return badges

  badges.set(products[0].product_id, { label: 'badge_best', tone: 'brand' })
  assigned.add(products[0].product_id)

  const remaining = products.filter((product) => !assigned.has(product.product_id))
  const topRated = [...remaining].sort((a, b) => b.rating - a.rating)[0]
  if (topRated) {
    badges.set(topRated.product_id, { label: 'badge_rated', tone: 'success' })
    assigned.add(topRated.product_id)
  }

  const cheapest = remaining
    .filter((product) => !assigned.has(product.product_id))
    .sort((a, b) => a.price - b.price)[0]
  if (cheapest) {
    badges.set(cheapest.product_id, { label: 'badge_value', tone: 'warning' })
  }
  return badges
}

/** One sentence for the whole set, built here so it follows the UI language. */
function stockSummary(items: InventoryItem[], t: I18n['t'], fill: I18n['fill']): string {
  const names = (status: InventoryItem['status']) =>
    items.filter((item) => item.status === status).map((item) => item.name).join(', ')

  if (items.some((item) => item.status === 'out_of_stock')) {
    return fill('inv_out', { names: names('out_of_stock') })
  }
  if (items.some((item) => item.status === 'low_stock')) {
    return fill('inv_low', { names: names('low_stock') })
  }
  return t('inv_all_in_stock')
}

/**
 * The presented products, with the stock the agent checked for them.
 *
 * Stock rides along with the cards rather than arriving as its own message: the agent
 * checks stock *before* it presents, so a separate bubble would show up before any
 * product is on screen and read as if it came out of nowhere.
 */
export default function ProductCards({
  products,
  stock,
  formatPrice,
}: {
  products: Product[]
  stock?: InventoryItem[]
  formatPrice: (amount: number) => string
}) {
  const { t, fill, category, tag } = useI18n()
  const badges = computeBadges(products)
  const stockOf = new Map((stock ?? []).map((item) => [item.product_id, item]))

  if (products.length === 0) {
    return <p className="pt-1 text-xs text-muted">{t('no_matches')}</p>
  }

  return (
    <div className="pt-1">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {products.map((product) => {
          const badge = badges.get(product.product_id)
          const status = stockOf.get(product.product_id)
          return (
            <div key={product.product_id} className="card flex flex-col p-3.5">
              <div className="flex items-start justify-between">
                <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-100 text-lg">
                  {CATEGORY_EMOJI[product.category] ?? '🛍️'}
                </span>
                {badge && (
                  <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${BADGE_TONES[badge.tone]}`}>
                    {t(badge.label)}
                  </span>
                )}
              </div>

              <p className="mt-2.5 line-clamp-2 text-sm font-semibold leading-snug">{product.name}</p>

              <div className="mt-1 flex items-center gap-1 text-xs text-muted">
                <Star size={12} className="fill-amber-400 text-amber-400" />
                <span className="font-medium text-ink">{product.rating}</span>
                <span className="text-faint">({product.rating_count})</span>
              </div>

              <p className="mt-2 text-base font-bold text-brand">{formatPrice(product.price)}</p>

              <div className="mt-2 flex flex-wrap gap-1">
                {product.tags.slice(0, 3).map((item) => (
                  <span key={item} className="chip !px-2 !py-0.5 !text-[10px]">
                    {tag(item)}
                  </span>
                ))}
              </div>

              <div className="mt-2 flex items-center justify-between gap-2">
                <p className="truncate text-[11px] text-faint">{category(product.category)}</p>
                {status && status.status !== 'in_stock' && (
                  <span
                    className={`shrink-0 rounded-full border px-1.5 py-0.5 text-[10px] font-medium ${STOCK_TONES[status.status]}`}
                  >
                    {status.status === 'out_of_stock'
                      ? t('stock_out')
                      : fill('stock_low', { n: status.stock })}
                  </span>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {stock && stock.length > 0 && (
        <p className="mt-2 text-[11px] text-muted">{stockSummary(stock, t, fill)}</p>
      )}
    </div>
  )
}
