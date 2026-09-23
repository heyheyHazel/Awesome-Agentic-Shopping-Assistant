import AgentHeader from './AgentHeader'

export default function MessageBubble({
  agent,
  text,
  time,
  streaming,
}: {
  agent: string
  text: string
  time: string
  streaming?: boolean
}) {
  return (
    <div className="flex gap-2.5">
      <div className="min-w-0 flex-1">
        <AgentHeader agent={agent} time={time} />
        <div className="mt-1.5 whitespace-pre-line rounded-2xl rounded-tl-md border border-line bg-card px-3.5 py-2.5 text-sm leading-relaxed shadow-sm">
          {text}
          {streaming && <span className="ml-0.5 inline-block h-4 w-[7px] translate-y-0.5 animate-pulse rounded-sm bg-brand" />}
        </div>
      </div>
    </div>
  )
}

export function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-2xl rounded-br-md bg-brand px-3.5 py-2.5 text-sm leading-relaxed text-white shadow-sm">
        {text}
      </div>
    </div>
  )
}
