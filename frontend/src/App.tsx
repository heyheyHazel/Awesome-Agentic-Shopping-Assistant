import { useEffect, useState } from 'react'
import { fetchUsers } from './api'
import { useI18n } from './i18n'
import { useAgentStream } from './hooks/useAgentStream'
import type { UserSummary } from './types'
import Header from './components/Header'
import AgentPanel from './components/AgentPanel'
import ChatPanel from './components/ChatPanel'
import ProfilePanel from './components/ProfilePanel'
import Footer from './components/Footer'

export default function App() {
  const { lang } = useI18n()
  const [users, setUsers] = useState<UserSummary[]>([])
  const [userId, setUserId] = useState('U001')
  const { feed, agentStates, profile, experiment, latencyMs, timings, isStreaming, send, newChat } =
    useAgentStream(userId, lang)

  useEffect(() => {
    fetchUsers()
      .then(setUsers)
      .catch(() => setUsers([]))
  }, [])

  const handleUserChange = (id: string) => {
    setUserId(id)
    newChat()
  }

  return (
    <div className="min-h-screen lg:h-dvh lg:overflow-hidden">
      <div className="mx-auto flex h-full max-w-[1560px] flex-col gap-4 px-4 py-4 lg:px-8">
        <Header />

        <main className="grid min-h-0 flex-1 items-stretch gap-5 lg:grid-cols-[300px_minmax(0,1fr)] lg:grid-rows-[minmax(0,1fr)] xl:grid-cols-[300px_minmax(0,1fr)_320px]">
          <div className="scroll-slim order-2 lg:order-1 lg:h-full lg:overflow-y-auto lg:pr-1">
            <AgentPanel states={agentStates} />
          </div>

          <div className="order-1 min-h-0 lg:order-2 lg:h-full">
            <ChatPanel feed={feed} isStreaming={isStreaming} onSend={send} onNewChat={newChat} />
          </div>

          <div className="scroll-slim order-3 lg:h-full lg:overflow-y-auto lg:pr-1">
            <ProfilePanel
              users={users}
              userId={userId}
              onUserChange={handleUserChange}
              profile={profile}
              experiment={experiment}
              latencyMs={latencyMs}
              timings={timings}
            />
          </div>
        </main>

        <Footer />
      </div>
    </div>
  )
}
