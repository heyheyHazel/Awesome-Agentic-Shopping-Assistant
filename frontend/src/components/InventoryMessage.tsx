import { useI18n } from '../i18n'
import type { InventoryItem } from '../types'
import AgentHeader from './AgentHeader'

const STATUS_STYLES: Record<InventoryItem['status'], string> = {
  in_stock: 'border-success/25 bg-success-soft text-success',
  low_stock: 'border-warning/25 bg-warning-soft text-warning',
  out_of_stock: 'border-danger/25 bg-red-50 text-danger',
}

export default function InventoryMessage({
  items,
  time,
}: {
  items: InventoryItem[]
  time: string
}) {
  const { t, fill } = useI18n()

  const statusLabel = (item: InventoryItem) => {
    if (item.status === 'in_stock') return fill('stock_in', { n: item.stock })
    if (item.status === 'low_stock') return fill('stock_low', { n: item.stock })
    return t('stock_out')
  }

  const names = (filter: InventoryItem['status']) =>
    items
      .filter((item) => item.status === filter)
      .map((item) => item.name)
      .join(', ')

  const outOfStock = items.some((item) => item.status === 'out_of_stock')
  const lowStock = items.some((item) => item.status === 'low_stock')
  const summary = outOfStock
    ? fill('inv_out', { names: names('out_of_stock') })
    : lowStock
      ? fill('inv_low', { names: names('low_stock') })
      : t('inv_all_in_stock')

  return (
    <div className="flex gap-2.5">
      <div className="min-w-0 flex-1">
        <AgentHeader agent="inventory" time={time} />
        <div className="mt-1.5 rounded-2xl rounded-tl-md border border-line bg-card px-3.5 py-2.5 shadow-sm">
          <p className="text-sm leading-relaxed">{summary}</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {items.map((item) => (
              <span
                key={item.product_id}
                className={`rounded-full border px-2.5 py-1 text-[11px] font-medium ${STATUS_STYLES[item.status]}`}
              >
                {item.name} · {statusLabel(item)}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
