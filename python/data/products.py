"""Demo product catalog served by the recall agent (prices in USD)."""

from models.schemas import Product

# (id, name, category, price, brand, stock, rating, reviews, tags)
_RAW: list[tuple] = [
    # ── Running Shoes ──────────────────────────────────────────────
    ("P001", "Air Flex Runner 3", "Running Shoes", 99.99, "AeroStep", 42, 4.6, 128, ["lightweight", "breathable", "daily training"]),
    ("P002", "Zoom Stride 2", "Running Shoes", 109.99, "StrideX", 35, 4.7, 96, ["cushioning", "durable", "daily training"]),
    ("P003", "PaceLift 1.0", "Running Shoes", 89.99, "PaceLift", 28, 4.4, 78, ["comfortable", "stable", "daily training"]),
    ("P004", "TrailBlazer GTX", "Running Shoes", 139.99, "NorthPeak", 12, 4.8, 64, ["trail", "waterproof", "grip"]),
    ("P005", "CloudStep Lite", "Running Shoes", 74.99, "CloudStep", 0, 4.2, 51, ["budget", "lightweight"]),
    # ── Sneakers ───────────────────────────────────────────────────
    ("P006", "Court Classic 90", "Sneakers", 84.99, "Courtly", 66, 4.5, 214, ["classic", "casual"]),
    ("P007", "Street Nova X", "Sneakers", 119.99, "NovaLab", 41, 4.3, 87, ["streetwear", "retro"]),
    # ── Audio ──────────────────────────────────────────────────────
    ("P008", "PulsePods Pro 3", "Audio", 199.99, "PulseSound", 120, 4.7, 342, ["noise cancelling", "wireless", "earbuds"]),
    ("P009", "EchoWave XM6", "Audio", 249.99, "EchoWave", 64, 4.8, 210, ["over-ear", "noise cancelling"]),
    ("P010", "BeatDrop Air", "Audio", 59.99, "BeatDrop", 200, 4.4, 480, ["budget", "earbuds"]),
    # ── Wearables ──────────────────────────────────────────────────
    ("P011", "PulseBand Fit 5", "Wearables", 149.99, "PulseWear", 88, 4.5, 156, ["fitness tracker", "heart rate"]),
    ("P012", "OrbitWatch Ultra", "Wearables", 399.99, "Orbit", 25, 4.6, 92, ["gps", "sports watch"]),
    # ── Beauty ─────────────────────────────────────────────────────
    ("P013", "Velvet Matte Lipstick No.21", "Beauty", 32.00, "Maison V", 300, 4.7, 512, ["lipstick", "long lasting"]),
    ("P014", "Glow Serum C+E", "Skincare", 48.00, "DermaLeaf", 150, 4.8, 268, ["serum", "brightening"]),
    ("P015", "Silk Finish Foundation", "Beauty", 39.00, "Maison V", 180, 4.4, 193, ["foundation", "natural finish"]),
    ("P016", "Hydra Mist Toner", "Skincare", 24.00, "DewPoint", 420, 4.6, 331, ["toner", "hydrating"]),
    ("P017", "Night Repair Cream", "Skincare", 58.00, "DermaLeaf", 90, 4.7, 147, ["anti-aging", "night cream"]),
    # ── Fashion ────────────────────────────────────────────────────
    ("P018", "Everyday Crew Tee", "Fashion", 29.99, "LoftBasic", 500, 4.5, 623, ["cotton", "basic"]),
    ("P019", "Floral Wrap Dress", "Fashion", 79.99, "Bella Rue", 60, 4.3, 88, ["dress", "summer"]),
    ("P020", "Heritage Trench Coat", "Fashion", 189.99, "Bella Rue", 18, 4.8, 54, ["coat", "classic"]),
    # ── Home & Kitchen ─────────────────────────────────────────────
    ("P021", "Stone Aroma Diffuser", "Home", 45.99, "HearthCo", 130, 4.6, 175, ["aroma", "home decor"]),
    ("P022", "Weighted Blanket 15lb", "Home", 69.99, "HearthCo", 55, 4.5, 142, ["sleep", "calming"]),
    ("P023", "Ceramic Dinner Set 16pc", "Kitchen", 129.99, "TableCraft", 40, 4.7, 96, ["kitchen", "ceramic"]),
    ("P024", "Pour-Over Coffee Kit", "Kitchen", 54.99, "BrewCraft", 75, 4.6, 118, ["coffee", "manual brew"]),
    # ── Sports & Outdoors ──────────────────────────────────────────
    ("P025", "Aero Yoga Mat 6mm", "Sports", 39.99, "ZenFlow", 210, 4.7, 356, ["yoga", "non-slip"]),
    ("P026", "PowerFlex Dumbbell 20kg", "Sports", 89.99, "IronCore", 30, 4.5, 71, ["strength", "adjustable"]),
    ("P027", "Summit Hiking Backpack 40L", "Outdoor", 119.99, "NorthPeak", 22, 4.8, 87, ["hiking", "waterproof"]),
    ("P028", "Campfire Folding Chair", "Outdoor", 44.99, "NorthPeak", 90, 4.4, 129, ["camping", "portable"]),
    # ── Accessories ────────────────────────────────────────────────
    ("P029", "Minimalist Card Wallet", "Accessories", 34.99, "LoftBasic", 160, 4.6, 203, ["leather", "slim"]),
    ("P030", "Polarized Sun Shades", "Accessories", 64.99, "VistaLine", 110, 4.5, 167, ["uv protection", "polarized"]),
    # ── Gaming / Food ──────────────────────────────────────────────
    ("P031", "PixelQuest Controller", "Gaming", 49.99, "PixelQuest", 140, 4.6, 231, ["wireless", "ergonomic"]),
    ("P032", "Single-Origin Coffee Beans 1kg", "Food", 28.99, "BrewCraft", 300, 4.8, 402, ["coffee", "arabica"]),
]

PRODUCTS: list[Product] = [
    Product(
        product_id=pid, name=name, category=category, price=price, brand=brand,
        stock=stock, rating=rating, rating_count=reviews, tags=tags,
    )
    for pid, name, category, price, brand, stock, rating, reviews, tags in _RAW
]
