import { Bot, Network, PackageCheck, PenLine } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

const AGENT_META: Record<string, { label: string; icon: LucideIcon; tone: string }> = {
  supervisor: { label: 'Supervisor Agent', icon: Network, tone: 'bg-brand-soft text-brand' },
  copywriting: { label: 'Copywriting Agent', icon: PenLine, tone: 'bg-vip-soft text-vip' },
  inventory: { label: 'Inventory Agent', icon: PackageCheck, tone: 'bg-success-soft text-success' },
  assistant: { label: 'Assistant', icon: Bot, tone: 'bg-slate-100 text-slate-600' },
}

export default function AgentHeader({ agent, time }: { agent: string; time: string }) {
  const meta = AGENT_META[agent] ?? AGENT_META.assistant
  const Icon = meta.icon
  return (
    <div className="flex items-center gap-2">
      <span className={`flex h-6 w-6 items-center justify-center rounded-md ${meta.tone}`}>
        <Icon size={13} />
      </span>
      <span className="text-xs font-semibold">{meta.label}</span>
      <span className="text-[11px] text-faint">{time}</span>
    </div>
  )
}
