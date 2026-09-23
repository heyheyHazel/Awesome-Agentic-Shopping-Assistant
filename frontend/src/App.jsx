import { useState } from 'react'
import ChatWindow from './components/ChatWindow'
import ChatInput from './components/ChatInput'
import { useChat } from './hooks/useChat'

export default function App() {
  const { messages, isStreaming, sendMessage } = useChat('user_001')

  return (
    <div className="app">
      <header className="app-header">
        <h1>AI 购物助手</h1>
        <p>告诉我你想买什么，我来帮你推荐</p>
      </header>
      <ChatWindow messages={messages} isStreaming={isStreaming} />
      <ChatInput onSend={sendMessage} disabled={isStreaming} />
    </div>
  )
}
