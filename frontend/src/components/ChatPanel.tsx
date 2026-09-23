import { useEffect, useRef, useState } from 'react'
import { MessagesSquare, RotateCcw, Send } from 'lucide-react'
import { useI18n } from '../i18n'
import type { StringKey } from '../i18n'
import type { FeedItem } from '../types'
import MessageBubble, { UserBubble } from './MessageBubble'
import ProductCards from './ProductCards'
import InventoryMessage from './InventoryMessage'

const SUGGESTIONS: StringKey[] = ['suggest_1', 'suggest_2', 'suggest_3']

function EmptyState({ onPick }: { onPick: (text: string) => void }) {
  const { t } = useI18n()
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
      <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-brand-soft text-2xl">🛍️</span>
      <p className="text-sm font-semibold">{t('empty_title')}</p>
      <p className="max-w-xs text-xs text-muted">{t('empty_desc')}</p>
      <div className="mt-2 flex flex-col gap-2">
        {SUGGESTIONS.map((key) => (
          <button
            key={key}
            onClick={() => onPick(t(key))}
            className="chip !text-xs transition hover:border-brand/40 hover:text-brand"
          >
            {t(key)}
          </button>
        ))}
      </div>
    </div>
  )
}

function ChatInput({ onSend, disabled }: { onSend: (text: string) => void; disabled: boolean }) {
  const { t } = useI18n()
  const [text, setText] = useState('')

  const submit = () => {
    if (!text.trim() || disabled) return
    onSend(text.trim())
    setText('')
  }

  return (
    <div className="border-t border-line p-3">
      <div className="flex items-center gap-2 rounded-xl border border-line bg-canvas/60 px-3 py-2 focus-within:border-brand/50 focus-within:ring-2 focus-within:ring-brand/10">
        <input
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              submit()
            }
          }}
          placeholder={t('input_placeholder')}
          className="flex-1 bg-transparent text-sm outline-none placeholder:text-faint"
        />
        <button
          onClick={submit}
          disabled={disabled || !text.trim()}
          className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand text-white transition hover:bg-brand/90 disabled:opacity-40"
          aria-label={t('send')}
        >
          <Send size={14} />
        </button>
      </div>
    </div>
  )
}

export default function ChatPanel({
  feed,
  isStreaming,
  onSend,
  onNewChat,
  formatPrice,
}: {
  feed: FeedItem[]
  isStreaming: boolean
  onSend: (text: string) => void
  onNewChat: () => void
  formatPrice: (amount: number) => string
}) {
  const { t } = useI18n()
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [feed])

  return (
    <section className="card flex h-full min-h-[540px] flex-col overflow-hidden lg:min-h-0">
      <header className="flex items-center justify-between border-b border-line px-4 py-3">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <MessagesSquare size={16} className="text-brand" />
          {t('chat_title')}
        </div>
        <div className="flex items-center gap-2">
          <span className="chip !text-[11px]">
            <span className={`h-1.5 w-1.5 rounded-full ${isStreaming ? 'animate-pulse bg-warning' : 'bg-success'}`} />
            {isStreaming ? t('sse_active') : t('sse_idle')}
          </span>
          <button onClick={onNewChat} className="chip !text-[11px] transition hover:text-ink">
            <RotateCcw size={11} />
            {t('new_chat')}
          </button>
        </div>
      </header>

      <div className="scroll-slim flex-1 space-y-4 overflow-y-auto px-4 py-5">
        {feed.length === 0 ? (
          <EmptyState onPick={onSend} />
        ) : (
          feed.map((item) => {
            switch (item.kind) {
              case 'user':
                return <UserBubble key={item.id} text={item.text} />
              case 'agent':
                return (
                  <MessageBubble
                    key={item.id}
                    agent={item.agent}
                    text={item.text}
                    time={item.time}
                    streaming={item.streaming}
                  />
                )
              case 'products':
                return <ProductCards key={item.id} products={item.products} formatPrice={formatPrice} />
              case 'inventory':
                return <InventoryMessage key={item.id} items={item.items} time={item.time} />
            }
          })
        )}
        <div ref={bottomRef} />
      </div>

      <ChatInput onSend={onSend} disabled={isStreaming} />
    </section>
  )
}
