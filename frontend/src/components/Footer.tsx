import { BrainCircuit, Radio, Target, UserRound } from 'lucide-react'

const FEATURES = [
  {
    icon: BrainCircuit,
    title: 'Multi-Agent Collaboration',
    desc: 'Supervisor plans, routes tasks and coordinates specialised agents.',
  },
  {
    icon: UserRound,
    title: 'Personalized Experience',
    desc: 'RFM clustering and user profiling drive every ranking.',
  },
  {
    icon: Target,
    title: 'Smarter Decisions',
    desc: 'Thompson Sampling powers data-driven A/B testing.',
  },
  {
    icon: Radio,
    title: 'Real-time Streaming',
    desc: 'SSE streams every agent step for instant feedback.',
  },
]

export default function Footer() {
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
              <p className="text-xs font-semibold">{feature.title}</p>
              <p className="mt-1 text-[11px] leading-relaxed text-muted">{feature.desc}</p>
            </div>
          </div>
        )
      })}
    </footer>
  )
}
