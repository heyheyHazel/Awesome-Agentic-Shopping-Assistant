import { Network, PenLine, PackageCheck, Sparkles } from 'lucide-react'
import { useI18n } from '../i18n'
import type { StringKey } from '../i18n'
import type { LucideIcon } from 'lucide-react'
import type { AgentStatus } from '../types'

interface AgentCard {
  key: string
  name: StringKey
  desc: StringKey
  icon: LucideIcon
}

const AGENTS: AgentCard[] = [
  { key: 'supervisor', name: 'agent_supervisor', desc: 'agent_supervisor_desc', icon: Network },
  { key: 'recommendation', name: 'agent_recommendation', desc: 'agent_recommendation_desc', icon: Sparkles },
  { key: 'copywriting', name: 'agent_copywriting', desc: 'agent_copywriting_desc', icon: PenLine },
  { key: 'inventory', name: 'agent_inventory', desc: 'agent_inventory_desc', icon: PackageCheck },
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

export default function AgentPanel({ states }: { states: Record<string, AgentStatus> }) {
  const { t } = useI18n()

  return (
    <aside className="flex flex-col gap-4">
      <p className="section-label px-1">{t('agents_label')}</p>
      <div className="flex flex-col gap-3">
        {AGENTS.map((agent) => {
          const status = states[agent.key] ?? 'idle'
          const Icon = agent.icon
          return (
            <div
              key={agent.key}
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
                <p className="flex-1 text-sm font-semibold leading-tight">{t(agent.name)}</p>
                <StatusDot status={status} />
              </div>
              <p className="mt-2 text-xs leading-relaxed text-muted">{t(agent.desc)}</p>
            </div>
          )
        })}
      </div>
    </aside>
  )
}
