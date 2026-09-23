import { Star } from 'lucide-react'
import type { Product } from '../types'

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
}

const BADGE_TONES: Record<string, string> = {
  brand: 'bg-brand-soft text-brand',
  success: 'bg-success-soft text-success',
  warning: 'bg-warning-soft text-warning',
}

/** First result = Best Match, next best rating = High Rated, cheapest remaining = Great Value. */
function computeBadges(products: Product[]) {
  const badges = new Map<string, { label: string; tone: string }>()
  const assigned = new Set<string>()
  if (products.length === 0) return badges

  badges.set(products[0].product_id, { label: 'Best Match', tone: 'brand' })
  assigned.add(products[0].product_id)

  const remaining = products.filter((product) => !assigned.has(product.product_id))
  const topRated = [...remaining].sort((a, b) => b.rating - a.rating)[0]
  if (topRated) {
    badges.set(topRated.product_id, { label: 'High Rated', tone: 'success' })
    assigned.add(topRated.product_id)
  }

  const cheapest = remaining
    .filter((product) => !assigned.has(product.product_id))
    .sort((a, b) => a.price - b.price)[0]
  if (cheapest) {
    badges.set(cheapest.product_id, { label: 'Great Value', tone: 'warning' })
  }
  return badges
}

export default function ProductCards({ products }: { products: Product[] }) {
  const badges = computeBadges(products)

  return (
    <div className="grid grid-cols-1 gap-3 pt-1 sm:grid-cols-2 xl:grid-cols-3">
      {products.map((product) => {
        const badge = badges.get(product.product_id)
        return (
          <div key={product.product_id} className="card flex flex-col p-3.5">
            <div className="flex items-start justify-between">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-100 text-lg">
                {CATEGORY_EMOJI[product.category] ?? '🛍️'}
              </span>
              {badge && (
                <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${BADGE_TONES[badge.tone]}`}>
                  {badge.label}
                </span>
              )}
            </div>

            <p className="mt-2.5 line-clamp-2 text-sm font-semibold leading-snug">{product.name}</p>

            <div className="mt-1 flex items-center gap-1 text-xs text-muted">
              <Star size={12} className="fill-amber-400 text-amber-400" />
              <span className="font-medium text-ink">{product.rating}</span>
              <span className="text-faint">({product.rating_count})</span>
            </div>

            <p className="mt-2 text-base font-bold text-brand">${product.price.toFixed(2)}</p>

            <div className="mt-2 flex flex-wrap gap-1">
              {product.tags.slice(0, 3).map((tag) => (
                <span key={tag} className="chip !px-2 !py-0.5 !text-[10px]">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}
