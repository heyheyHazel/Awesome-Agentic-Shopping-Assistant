import { Bot } from 'lucide-react'

const TECH = ['LangGraph', 'Multi-Agent', 'FastAPI', 'React 19', 'TypeScript']

export default function Header() {
  return (
    <header className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
      <div className="flex items-start gap-3">
        <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-brand to-vip text-white shadow-sm">
          <Bot size={22} />
        </span>
        <div>
          <h1 className="text-xl font-semibold tracking-tight lg:text-2xl">Multi-Agent E-commerce Assistant</h1>
          <p className="mt-1 max-w-xl text-sm text-muted">
            Intelligent shopping with multi-agent orchestration, personalization, and real-time responses.
          </p>
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5 lg:justify-end">
        {TECH.map((item) => (
          <span key={item} className="chip">
            {item}
          </span>
        ))}
      </div>
    </header>
  )
}
