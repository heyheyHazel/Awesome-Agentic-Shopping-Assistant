/** Format a catalog amount with the active currency (zero-decimal currencies stay clean). */
export function formatPrice(amount: number, currency: string, symbol: string): string {
  const digits =
    currency === 'CNY' || currency === 'JPY'
      ? { maximumFractionDigits: 2 }
      : { minimumFractionDigits: 2, maximumFractionDigits: 2 }
  return `${symbol}${amount.toLocaleString('en-US', digits)}`
}
