export default function ProductCard({ product }) {
  return (
    <div className="product-card">
      <div className="product-card-image">
        {product.category}
      </div>
      <div className="product-card-name" title={product.name}>
        {product.name}
      </div>
      <div className="product-card-price">
        ¥{product.price}
      </div>
      <div className="product-card-meta">
        <span className="product-card-tag">{product.brand}</span>
        {product.tags?.slice(0, 2).map((tag) => (
          <span key={tag} className="product-card-tag">{tag}</span>
        ))}
      </div>
      {product.marketing_copy && (
        <div className="product-card-copy">{product.marketing_copy}</div>
      )}
    </div>
  )
}
