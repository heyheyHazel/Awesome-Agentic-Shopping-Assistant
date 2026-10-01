import type { LucideIcon } from 'lucide-react'
import { Bot } from 'lucide-react'
import { useI18n } from '../i18n'
import type { StringKey } from '../i18n'

const AGENT_META: Record<string, { label: StringKey; icon: LucideIcon; tone: string }> = {
  assistant: { label: 'agent_assistant', icon: Bot, tone: 'bg-slate-100 text-slate-600' },
}

export default function AgentHeader({ agent, time }: { agent: string; time: string }) {
  const { t } = useI18n()
  const meta = AGENT_META[agent] ?? AGENT_META.assistant
  const Icon = meta.icon
  return (
    <div className="flex items-center gap-2">
      <span className={`flex h-6 w-6 items-center justify-center rounded-md ${meta.tone}`}>
        <Icon size={13} />
      </span>
      <span className="text-xs font-semibold">{t(meta.label)}</span>
      <span className="text-[11px] text-faint">{time}</span>
    </div>
  )
}
