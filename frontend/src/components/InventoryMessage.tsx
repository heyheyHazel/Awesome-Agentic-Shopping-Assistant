import AgentHeader from './AgentHeader'
import type { InventoryItem } from '../types'

const STATUS_STYLES: Record<InventoryItem['status'], string> = {
  in_stock: 'border-success/25 bg-success-soft text-success',
  low_stock: 'border-warning/25 bg-warning-soft text-warning',
  out_of_stock: 'border-danger/25 bg-red-50 text-danger',
}

function statusLabel(item: InventoryItem): string {
  if (item.status === 'in_stock') return `In Stock (${item.stock})`
  if (item.status === 'low_stock') return `Low Stock (${item.stock})`
  return 'Sold Out'
}

export default function InventoryMessage({
  items,
  summary,
  time,
}: {
  items: InventoryItem[]
  summary: string
  time: string
}) {
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
