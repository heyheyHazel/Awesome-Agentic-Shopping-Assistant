import { useCallback, useMemo, useState } from 'react'
import { ChevronDown, Shuffle, Timer, Trophy } from 'lucide-react'
import { useI18n } from '../i18n'
import type { ReactNode } from 'react'
import type { ExperimentInfo, ProfileResponse, UserSummary } from '../types'

// The roster holds thousands of shoppers, so the picker pages through them.
const USER_BATCH_SIZE = 10

/**
 * Deterministic Fisher-Yates: the same list and seed always give the same order.
 * Seeded rather than random so the page is stable across re-renders (including
 * React's double render in development) and only changes when a new seed is set.
 */
function seededShuffle<T>(items: T[], seed: number): T[] {
  const copy = [...items]
  let state = seed * 2654435761 + 1
  const next = () => {
    state = (state * 1103515245 + 12345) % 2147483648
    return state / 2147483648
  }
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(next() * (i + 1))
    ;[copy[i], copy[j]] = [copy[j], copy[i]]
  }
  return copy
}

/**
 * One page of shoppers for the picker: a stable random sample of the roster, with
 * the current selection forced to the front so the control always has a valid
 * value without disturbing the rest of the page.
 */
function useUserBatch(users: UserSummary[], selectedId: string) {
  const [seed, setSeed] = useState(0)

  const sample = useMemo(
    () => seededShuffle(users, seed).slice(0, USER_BATCH_SIZE),
    [users, seed],
  )

  const batch = useMemo(() => {
    if (sample.some((user) => user.user_id === selectedId)) return sample
    const selected = users.find((user) => user.user_id === selectedId)
    return selected ? [selected, ...sample.slice(0, USER_BATCH_SIZE - 1)] : sample
  }, [sample, users, selectedId])

  const reshuffle = useCallback(() => setSeed((current) => current + 1), [])
  return { batch, reshuffle }
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
  formatPrice,
}: {
  users: UserSummary[]
  userId: string
  onUserChange: (id: string) => void
  profile: ProfileResponse | null
  experiment: ExperimentInfo | null
  latencyMs: number | null
  timings: Record<string, number> | null
  formatPrice: (amount: number) => string
}) {
  const { t, fill, segment, category } = useI18n()
  const [expanded, setExpanded] = useState(false)
  const person = profile?.profile
  const { batch, reshuffle } = useUserBatch(users, userId)
  const slowest = timings
    ? Object.entries(timings)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 4)
    : []

  return (
    <aside className="flex flex-col gap-4">
      <div className="card p-4">
        <div className="flex items-center justify-between gap-2">
          <p className="section-label shrink-0">{t('profile_title')}</p>
          <div className="flex min-w-0 items-center gap-0.5 rounded-full border border-line bg-card p-0.5 pl-2">
            <select
              value={userId}
              onChange={(event) => onUserChange(event.target.value)}
              className="w-[104px] truncate bg-transparent text-[11px] font-semibold text-muted outline-none"
            >
              {batch.map((user) => (
                <option key={user.user_id} value={user.user_id}>
                  {user.tier === 'VIP' ? `${user.name} · VIP` : user.name}
                </option>
              ))}
            </select>
            <button
              type="button"
              title={t('users_shuffle')}
              aria-label={t('users_shuffle')}
              onClick={reshuffle}
              className="shrink-0 rounded-full p-1 text-faint transition hover:bg-slate-100 hover:text-brand"
            >
              <Shuffle size={12} />
            </button>
          </div>
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
                  value={formatPrice(person.rfm.lifetime_value)}
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
                  value={`${formatPrice(person.price_range[0])} – ${formatPrice(person.price_range[1])}`}
                />
              </div>
            )}
          </>
        ) : (
          <p className="mt-3 text-xs text-faint">{t('waiting_backend')}</p>
        )}
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
