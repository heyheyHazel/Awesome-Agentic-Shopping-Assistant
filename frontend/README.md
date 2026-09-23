# Frontend — Multi-Agent Shopping Assistant

React 19 + TypeScript + Vite + Tailwind CSS dashboard for the multi-agent backend.

```bash
npm install
npm run dev      # http://localhost:5173 (proxies /api to :8000)
npm run build    # type-check + production bundle
npm run lint     # oxlint
```

Layout: agent status panel (left) · live SSE conversation (center) · shopper profile, RFM clustering and A/B panel (right).
All live data comes from the SSE stream of `POST /api/v1/chat` — see `src/hooks/useAgentStream.ts`.
