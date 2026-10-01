import { Bot, PackageCheck, Search, Sparkles, UserRound } from 'lucide-react'
import { useI18n } from '../i18n'
import type { StringKey } from '../i18n'
import type { LucideIcon } from 'lucide-react'
import type { AgentStatus } from '../types'

interface ToolCard {
  key: string
  name: StringKey
  desc: StringKey
  icon: LucideIcon
}

/**
 * The agent is one tool-calling loop; everything else it can reach is a deterministic
 * tool. The keys are the tool names the backend reports, so the panel lights up from
 * real calls rather than from a scripted sequence.
 */
const PANEL: ToolCard[] = [
  { key: 'assistant', name: 'tool_assistant', desc: 'tool_assistant_desc', icon: Bot },
  { key: 'search_catalog', name: 'tool_search', desc: 'tool_search_desc', icon: Search },
  { key: 'get_shopper_profile', name: 'tool_profile', desc: 'tool_profile_desc', icon: UserRound },
  { key: 'check_inventory', name: 'tool_inventory', desc: 'tool_inventory_desc', icon: PackageCheck },
  { key: 'present_recommendation', name: 'tool_present', desc: 'tool_present_desc', icon: Sparkles },
]

function StatusDot({ status }: { status: AgentStatus }) {
  if (status === 'running') {
    return (
      <span className="relative flex h-2.5 w-2.5">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-warning opacity-60" />
        <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-warning" />
      </span>
    )
  }
  return (
    <span
      className={`inline-flex h-2.5 w-2.5 rounded-full ${status === 'done' ? 'bg-success' : 'bg-slate-300'}`}
    />
  )
}

export default function AgentPanel({
  states,
  order,
}: {
  states: Record<string, AgentStatus>
  order: string[]
}) {
  const { t } = useI18n()

  // Cards follow the order the tools were actually called in this turn, so the panel
  // reads as the sequence of steps rather than a fixed list the calls do not match.
  // Anything not called yet keeps its default slot, underneath.
  const rank = (key: string) => {
    const index = order.indexOf(key)
    return index === -1 ? order.length + PANEL.findIndex((entry) => entry.key === key) : index
  }
  const ordered = [...PANEL].sort((a, b) => rank(a.key) - rank(b.key))

  return (
    <aside className="flex flex-col gap-4">
      <p className="section-label px-1">{t('tools_label')}</p>
      <div className="flex flex-col gap-3">
        {ordered.map((tool) => {
          const status = states[tool.key] ?? 'idle'
          const step = order.indexOf(tool.key)
          const Icon = tool.icon
          return (
            <div
              key={tool.key}
              className={`card p-3.5 transition ${
                status === 'running' ? 'border-warning/40 ring-2 ring-warning/15' : ''
              }`}
            >
              <div className="flex items-center gap-2.5">
                <span
                  className={`flex h-8 w-8 items-center justify-center rounded-lg ${
                    status === 'running'
                      ? 'bg-warning-soft text-warning'
                      : status === 'done'
                        ? 'bg-success-soft text-success'
                        : 'bg-slate-100 text-slate-500'
                  }`}
                >
                  <Icon size={16} />
                </span>
                <p className="flex-1 text-sm font-semibold leading-tight">{t(tool.name)}</p>
                {step >= 0 && <span className="text-[10px] font-semibold text-faint">{step + 1}</span>}
                <StatusDot status={status} />
              </div>
              <p className="mt-2 text-xs leading-relaxed text-muted">{t(tool.desc)}</p>
            </div>
          )
        })}
      </div>
    </aside>
  )
}
