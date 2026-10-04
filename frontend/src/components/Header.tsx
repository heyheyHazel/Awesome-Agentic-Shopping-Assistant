import { Bot, Globe } from 'lucide-react'
import { useI18n } from '../i18n'
import type { Lang } from '../i18n'

const TECH = ['LangGraph', 'Tool Calling', 'FastAPI', 'React 19', 'TypeScript']

export default function Header() {
  const { lang, setLang, t } = useI18n()

  return (
    <header className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
      <div className="flex items-center gap-3">
        <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-brand to-vip text-white shadow-sm">
          <Bot size={22} />
        </span>
        <h1 className="text-xl font-semibold tracking-tight lg:text-2xl">Awesome Agentic Shopping Assistant</h1>
      </div>

      <div className="flex flex-wrap items-center gap-3 lg:justify-end">
        <div className="flex flex-wrap gap-1.5">
          {TECH.map((item) => (
            <span key={item} className="chip">
              {item}
            </span>
          ))}
        </div>

        <div className="flex items-center gap-1 rounded-full border border-line bg-card p-0.5 pl-2">
          <Globe size={13} className="text-faint" />
          {(['en', 'zh'] as Lang[]).map((code) => (
            <button
              key={code}
              onClick={() => setLang(code)}
              aria-pressed={lang === code}
              className={`rounded-full px-2.5 py-1 text-[11px] font-semibold transition ${
                lang === code ? 'bg-brand text-white' : 'text-muted hover:text-ink'
              }`}
            >
              {code === 'en' ? t('lang_en') : t('lang_zh')}
            </button>
          ))}
        </div>
      </div>
    </header>
  )
}
