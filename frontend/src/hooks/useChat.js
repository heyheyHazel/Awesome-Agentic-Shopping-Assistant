import { useState, useCallback, useRef } from 'react'

export function useChat(userId) {
  const [messages, setMessages] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef(null)

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || isStreaming) return

    const userMessage = { role: 'user', content: text, products: [] }
    const assistantMessage = { role: 'assistant', content: '', products: [], isStreaming: true }

    setMessages((prev) => [...prev, userMessage, assistantMessage])
    setIsStreaming(true)

    const allMessages = [...messages, userMessage].map(({ role, content }) => ({ role, content }))

    try {
      const controller = new AbortController()
      abortRef.current = controller

      const response = await fetch('/api/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, messages: allMessages }),
        signal: controller.signal,
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let currentEvent = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7)
          } else if (line.startsWith('data: ')) {
            const dataStr = line.slice(6)
            try {
              const data = JSON.parse(dataStr)
              switch (currentEvent) {
                case 'token':
                  setMessages((prev) => {
                    const updated = [...prev]
                    const last = { ...updated[updated.length - 1] }
                    last.content += data.content
                    updated[updated.length - 1] = last
                    return updated
                  })
                  break
                case 'products':
                  setMessages((prev) => {
                    const updated = [...prev]
                    const last = { ...updated[updated.length - 1] }
                    last.products = data.products || []
                    updated[updated.length - 1] = last
                    return updated
                  })
                  break
                case 'done':
                  break
              }
            } catch (e) {
              // skip malformed JSON
            }
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setMessages((prev) => {
          const updated = [...prev]
          const last = { ...updated[updated.length - 1] }
          last.content += '\n\n[连接出错，请重试]'
          updated[updated.length - 1] = last
          return updated
        })
      }
    } finally {
      setMessages((prev) => {
        const updated = [...prev]
        const last = { ...updated[updated.length - 1] }
        last.isStreaming = false
        updated[updated.length - 1] = last
        return updated
      })
      setIsStreaming(false)
      abortRef.current = null
    }
  }, [messages, isStreaming, userId])

  return { messages, isStreaming, sendMessage }
}
