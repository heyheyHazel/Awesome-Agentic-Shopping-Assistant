import ProductCard from './ProductCard'

export default function ProductCardList({ products }) {
  return (
    <div className="product-card-list">
      {products.map((product) => (
        <ProductCard key={product.product_id} product={product} />
      ))}
    </div>
  )
}
