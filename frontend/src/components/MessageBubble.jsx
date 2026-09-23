import ProductCardList from './ProductCardList'

export default function MessageBubble({ message }) {
  const { role, content, products } = message

  return (
    <div className={`message ${role}`}>
      <div className="message-avatar">
        {role === 'user' ? '👤' : '🤖'}
      </div>
      <div className="message-content">
        {content}
        {products && products.length > 0 && (
          <ProductCardList products={products} />
        )}
      </div>
    </div>
  )
}
