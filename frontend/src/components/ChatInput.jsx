import { useState } from 'react'

export default function ChatInput({ onSend, disabled }) {
  const [text, setText] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (text.trim() && !disabled) {
      onSend(text.trim())
      setText('')
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <form className="chat-input-container" onSubmit={handleSubmit}>
      <input
        className="chat-input"
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="输入你想买的商品，如「推荐口红」..."
        disabled={disabled}
      />
      <button className="chat-send-btn" type="submit" disabled={disabled || !text.trim()}>
        发送
      </button>
    </form>
  )
}
