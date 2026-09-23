import { useEffect, useState } from 'react'
import { fetchUsers } from './api'
import { useAgentStream } from './hooks/useAgentStream'
import type { UserSummary } from './types'
import Header from './components/Header'
import AgentPanel from './components/AgentPanel'
import ChatPanel from './components/ChatPanel'
import ProfilePanel from './components/ProfilePanel'
import Footer from './components/Footer'

export default function App() {
  const [users, setUsers] = useState<UserSummary[]>([])
  const [userId, setUserId] = useState('U001')
  const { feed, agentStates, profile, experiment, latencyMs, timings, isStreaming, send, newChat } =
    useAgentStream(userId)

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
    <div className="min-h-screen">
      <div className="mx-auto flex max-w-[1560px] flex-col gap-5 px-4 py-6 lg:px-8">
        <Header />

        <main className="grid items-start gap-5 lg:grid-cols-[300px_minmax(0,1fr)] xl:grid-cols-[300px_minmax(0,1fr)_320px]">
          <div className="order-2 lg:order-1">
            <AgentPanel states={agentStates} />
          </div>

          <div className="order-1 lg:order-2">
            <ChatPanel feed={feed} isStreaming={isStreaming} onSend={send} onNewChat={newChat} />
          </div>

          <div className="order-3 xl:sticky xl:top-6">
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
