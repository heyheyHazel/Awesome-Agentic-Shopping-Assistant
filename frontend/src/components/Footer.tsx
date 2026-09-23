import { BrainCircuit, Radio, Target, UserRound } from 'lucide-react'
import { useI18n } from '../i18n'
import type { StringKey } from '../i18n'

const FEATURES: { icon: typeof BrainCircuit; title: StringKey; desc: StringKey }[] = [
  { icon: BrainCircuit, title: 'feature_collab_title', desc: 'feature_collab_desc' },
  { icon: UserRound, title: 'feature_personal_title', desc: 'feature_personal_desc' },
  { icon: Target, title: 'feature_decision_title', desc: 'feature_decision_desc' },
  { icon: Radio, title: 'feature_stream_title', desc: 'feature_stream_desc' },
]

export default function Footer() {
  const { t } = useI18n()

  return (
    <footer className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {FEATURES.map((feature) => {
        const Icon = feature.icon
        return (
          <div key={feature.title} className="card flex items-start gap-3 p-4">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-brand-soft text-brand">
              <Icon size={15} />
            </span>
            <div>
              <p className="text-xs font-semibold">{t(feature.title)}</p>
              <p className="mt-1 text-[11px] leading-relaxed text-muted">{t(feature.desc)}</p>
            </div>
          </div>
        )
      })}
    </footer>
  )
}
