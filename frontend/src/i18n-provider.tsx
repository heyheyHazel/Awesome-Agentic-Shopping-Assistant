import { useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { I18nContext, buildI18n, initialLang } from './i18n'
import type { Lang } from './i18n'

export default function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(initialLang)

  useEffect(() => {
    window.localStorage.setItem('lang', lang)
    document.documentElement.lang = lang
  }, [lang])

  const value = useMemo(() => buildI18n(lang, setLang), [lang])

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}
