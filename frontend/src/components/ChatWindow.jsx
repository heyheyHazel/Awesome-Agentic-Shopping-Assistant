import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'

export default function ChatWindow({ messages, isStreaming }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="chat-window">
      {messages.length === 0 && (
        <div style={{ textAlign: 'center', color: '#999', marginTop: '40px' }}>
          <p style={{ fontSize: '24px', marginBottom: '8px' }}>👋</p>
          <p>你好！我是 AI 购物助手</p>
          <p style={{ fontSize: '13px', marginTop: '4px' }}>试试说「推荐一款口红」或「200元以下的面膜」</p>
        </div>
      )}
      {messages.map((msg, idx) => (
        <MessageBubble key={idx} message={msg} />
      ))}
      {isStreaming && messages[messages.length - 1]?.content === '' && (
        <div className="message assistant">
          <div className="message-avatar">🤖</div>
          <div className="message-content">
            <div className="typing-indicator">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  )
}
