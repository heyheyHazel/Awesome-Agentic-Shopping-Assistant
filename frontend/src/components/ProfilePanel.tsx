import { useState } from 'react'
import { ChevronDown, Timer, Trophy } from 'lucide-react'
import { useI18n } from '../i18n'
import type { ReactNode } from 'react'
import type { ExperimentInfo, ProfileResponse, UserSummary } from '../types'

const SEGMENT_DOTS: Record<string, string> = {
  Champions: 'bg-amber-400',
  Loyal: 'bg-brand',
  Potential: 'bg-teal-500',
  'At Risk': 'bg-warning',
  New: 'bg-slate-400',
}

const SEGMENT_BADGES: Record<string, string> = {
  Champions: 'bg-amber-50 text-amber-600',
  Loyal: 'bg-brand-soft text-brand',
  Potential: 'bg-teal-50 text-teal-600',
  'At Risk': 'bg-warning-soft text-warning',
  New: 'bg-slate-100 text-slate-500',
}

function initials(name: string): string {
  return name
    .split(' ')
    .map((part) => part[0])
    .join('')
    .slice(0, 2)
    .toUpperCase()
}

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-3 border-t border-line/70 pt-2 first:border-0 first:pt-0">
      <span className="shrink-0 text-xs text-muted">{label}</span>
      <span className="max-w-[64%] text-right text-xs font-semibold">{value}</span>
    </div>
  )
}

function RfmRow({ label, tone, active, currentLabel }: { label: string; tone: string; active: boolean; currentLabel: string }) {
  return (
    <div
      className={`flex items-center gap-2 rounded-lg px-2 py-1.5 text-xs ${
        active ? 'bg-canvas font-semibold ring-1 ring-line' : 'text-muted'
      }`}
    >
      <span className={`h-2 w-2 rounded-full ${tone}`} />
      {label}
      {active && <span className="ml-auto text-[10px] font-semibold text-brand">{currentLabel}</span>}
    </div>
  )
}

function AbCard({ experiment }: { experiment: ExperimentInfo | null }) {
  const { t, fill } = useI18n()
  return (
    <div className="card p-4">
      <p className="section-label">{t('ab_title')}</p>
      {experiment ? (
        <>
          <div className="mt-1 flex items-center justify-between text-[11px] text-faint">
            <span>{experiment.name}</span>
            <span>{t('ab_conversion')}</span>
          </div>
          <div className="mt-3 space-y-3">
            {experiment.variants.map((variant) => (
              <div key={variant.name}>
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium">
                    {variant.name} <span className="text-faint">({variant.label})</span>
                    {variant.name === experiment.variant && (
                      <span className="ml-1.5 rounded-full bg-brand-soft px-1.5 py-0.5 text-[10px] font-semibold text-brand">
                        {t('ab_bucket')}
                      </span>
                    )}
                  </span>
                  <span className="font-semibold">{variant.conversion_rate}%</span>
                </div>
                <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className={`h-full rounded-full ${variant.name === experiment.winner ? 'bg-success' : 'bg-brand'}`}
                    style={{ width: `${variant.conversion_rate}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-3 flex items-center gap-1.5 rounded-lg bg-success-soft px-2.5 py-1.5 text-[11px] font-semibold text-success">
            <Trophy size={12} />
            {fill('ab_winner', { name: experiment.winner })}
          </div>
        </>
      ) : (
        <p className="mt-2 text-xs text-faint">{t('waiting_backend')}</p>
      )}
    </div>
  )
}

export default function ProfilePanel({
  users,
  userId,
  onUserChange,
  profile,
  experiment,
  latencyMs,
  timings,
}: {
  users: UserSummary[]
  userId: string
  onUserChange: (id: string) => void
  profile: ProfileResponse | null
  experiment: ExperimentInfo | null
  latencyMs: number | null
  timings: Record<string, number> | null
}) {
  const { t, fill, segment, category } = useI18n()
  const [expanded, setExpanded] = useState(false)
  const person = profile?.profile
  const slowest = timings
    ? Object.entries(timings)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 4)
    : []

  return (
    <aside className="flex flex-col gap-4">
      <div className="card p-4">
        <div className="flex items-center justify-between">
          <p className="section-label">{t('profile_title')}</p>
          <select
            value={userId}
            onChange={(event) => onUserChange(event.target.value)}
            className="rounded-lg border border-line bg-card px-2 py-1 text-[11px] font-medium text-muted outline-none focus:border-brand/50"
          >
            {users.map((user) => (
              <option key={user.user_id} value={user.user_id}>
                {user.name}
              </option>
            ))}
          </select>
        </div>

        {person ? (
          <>
            <div className="mt-3 flex items-center gap-3">
              <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-brand to-vip text-sm font-bold text-white">
                {initials(person.name)}
              </span>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <p className="truncate text-sm font-semibold">{person.name}</p>
                  {person.tier === 'VIP' && (
                    <span className="rounded-full bg-vip-soft px-2 py-0.5 text-[10px] font-bold text-vip">VIP</span>
                  )}
                </div>
                <p className="truncate text-xs text-muted">{person.email}</p>
              </div>
              <span
                className={`ml-auto shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                  SEGMENT_BADGES[person.segment] ?? 'bg-slate-100 text-slate-500'
                }`}
              >
                {segment(person.segment)}
              </span>
            </div>

            <button
              onClick={() => setExpanded((value) => !value)}
              aria-expanded={expanded}
              className="mt-3 flex w-full items-center justify-between rounded-lg border border-line px-2.5 py-1.5 text-[11px] font-medium text-muted transition hover:border-brand/40 hover:text-brand"
            >
              {expanded ? t('hide_full') : t('show_full')}
              <ChevronDown size={13} className={`transition-transform ${expanded ? 'rotate-180' : ''}`} />
            </button>

            {expanded && (
              <div className="mt-3 space-y-2">
                <Row
                  label={t('row_segment')}
                  value={
                    <span className={`rounded-full px-2 py-0.5 ${SEGMENT_BADGES[person.segment] ?? ''}`}>
                      {segment(person.segment)}
                    </span>
                  }
                />
                <Row label={t('row_recency')} value={fill('recency_value', { n: person.rfm.recency_days })} />
                <Row label={t('row_frequency')} value={fill('frequency_value', { n: person.rfm.orders })} />
                <Row
                  label={t('row_monetary')}
                  value={`$${person.rfm.lifetime_value.toLocaleString('en-US', {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}`}
                />
                <Row
                  label={t('row_rfm_score')}
                  value={
                    <span className="block">
                      {person.rfm.overall.toFixed(2)}
                      <span className="block text-[10px] font-normal text-faint">
                        R {person.rfm.recency.toFixed(2)} · F {person.rfm.frequency.toFixed(2)} · M{' '}
                        {person.rfm.monetary.toFixed(2)}
                      </span>
                    </span>
                  }
                />
                <Row
                  label={t('row_preferred')}
                  value={person.preferred_categories.map((item) => category(item)).join(', ') || '—'}
                />
                <Row
                  label={t('row_budget')}
                  value={`$${person.price_range[0].toFixed(0)} – $${person.price_range[1].toFixed(0)}`}
                />
              </div>
            )}
          </>
        ) : (
          <p className="mt-3 text-xs text-faint">{t('waiting_backend')}</p>
        )}
      </div>

      <div className="card p-4">
        <p className="section-label">{t('rfm_clustering')}</p>
        <div className="mt-2 space-y-1">
          {(profile?.segments ?? Object.keys(SEGMENT_DOTS)).map((value) => (
            <RfmRow
              key={value}
              label={segment(value)}
              tone={SEGMENT_DOTS[value] ?? 'bg-slate-400'}
              active={person?.segment === value}
              currentLabel={t('current')}
            />
          ))}
        </div>
      </div>

      <AbCard experiment={experiment} />

      <div className="card flex items-center justify-between p-4">
        <div>
          <p className="section-label">{t('response_time')}</p>
          <p className="mt-1 text-xl font-bold">
            {latencyMs !== null ? `${(latencyMs / 1000).toFixed(1)}s` : '—'}
          </p>
          {slowest.length > 0 && (
            <div className="mt-2 space-y-0.5">
              {slowest.map(([name, ms]) => (
                <p key={name} className="text-[10px] text-faint">
                  {name} · {(ms / 1000).toFixed(1)}s
                </p>
              ))}
            </div>
          )}
        </div>
        <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-soft text-brand">
          <Timer size={18} />
        </span>
      </div>
    </aside>
  )
}
