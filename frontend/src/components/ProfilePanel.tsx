import { Timer, Trophy } from 'lucide-react'
import type { ReactNode } from 'react'
import type { ExperimentInfo, ProfileResponse, UserSummary } from '../types'

const SEGMENT_DOTS: Record<string, string> = {
  Champions: 'bg-amber-400',
  Loyal: 'bg-brand',
  Potential: 'bg-teal-500',
  'At Risk': 'bg-warning',
  New: 'bg-slate-400',
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
    <div className="flex items-center justify-between border-t border-line/70 pt-2 first:border-0 first:pt-0">
      <span className="text-xs text-muted">{label}</span>
      <span className="text-xs font-semibold">{value}</span>
    </div>
  )
}

function RfmRow({ label, tone, active }: { label: string; tone: string; active: boolean }) {
  return (
    <div
      className={`flex items-center gap-2 rounded-lg px-2 py-1.5 text-xs ${
        active ? 'bg-canvas font-semibold ring-1 ring-line' : 'text-muted'
      }`}
    >
      <span className={`h-2 w-2 rounded-full ${tone}`} />
      {label}
      {active && <span className="ml-auto text-[10px] font-semibold text-brand">current</span>}
    </div>
  )
}

function AbCard({ experiment }: { experiment: ExperimentInfo | null }) {
  return (
    <div className="card p-4">
      <p className="section-label">A/B Testing · Thompson Sampling</p>
      {experiment ? (
        <>
          <div className="mt-1 flex items-center justify-between text-[11px] text-faint">
            <span>{experiment.name}</span>
            <span>Conversion Rate</span>
          </div>
          <div className="mt-3 space-y-3">
            {experiment.variants.map((variant) => (
              <div key={variant.name}>
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium">
                    {variant.name} <span className="text-faint">({variant.label})</span>
                    {variant.name === experiment.variant && (
                      <span className="ml-1.5 rounded-full bg-brand-soft px-1.5 py-0.5 text-[10px] font-semibold text-brand">
                        your bucket
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
            Winner: Variant {experiment.winner}
          </div>
        </>
      ) : (
        <p className="mt-2 text-xs text-faint">Waiting for the backend…</p>
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
          <p className="section-label">User Profile</p>
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
              <span className="flex h-11 w-11 items-center justify-center rounded-full bg-gradient-to-br from-brand to-vip text-sm font-bold text-white">
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
            </div>

            <div className="mt-4 space-y-2">
              <Row
                label="RFM Segment"
                value={<span className="rounded-full bg-amber-50 px-2 py-0.5 text-amber-600">{person.segment}</span>}
              />
              <Row label="Recency" value={`${person.rfm.recency_days} days ago`} />
              <Row label="Frequency" value={`${person.rfm.orders} orders`} />
              <Row
                label="Monetary"
                value={`$${person.rfm.lifetime_value.toLocaleString('en-US', {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}`}
              />
            </div>
          </>
        ) : (
          <p className="mt-3 text-xs text-faint">Waiting for the backend…</p>
        )}
      </div>

      <div className="card p-4">
        <p className="section-label">RFM Clustering</p>
        <div className="mt-2 space-y-1">
          {(profile?.segments ?? Object.keys(SEGMENT_DOTS)).map((segment) => (
            <RfmRow
              key={segment}
              label={segment}
              tone={SEGMENT_DOTS[segment] ?? 'bg-slate-400'}
              active={person?.segment === segment}
            />
          ))}
        </div>
      </div>

      <AbCard experiment={experiment} />

      <div className="card flex items-center justify-between p-4">
        <div>
          <p className="section-label">Response Time</p>
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
