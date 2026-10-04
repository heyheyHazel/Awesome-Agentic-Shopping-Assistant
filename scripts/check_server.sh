#!/usr/bin/env bash
# Prove the interactive app works over real HTTP, with no API key and no GPU.
#
# Starts the canned model, starts the real server against the real catalogue, and
# checks the HTTP surface plus a full SSE conversation: session, experiment, tool
# events, product cards, the reply and done.
#
# Usage:  bash scripts/check_server.sh
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
# Prefer the project venv when there is one, otherwise whatever python is on
# PATH — CI installs into the runner's interpreter and has no .venv.
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
  else
    PYTHON="$(command -v python3 || command -v python)"
  fi
fi
PORT="${PORT:-8077}"
STUB_PORT="${STUB_PORT:-9099}"
FAILED=0

pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; FAILED=$((FAILED + 1)); }

APP_PID=""
STUB_PID=""
cleanup() {
  [ -n "$APP_PID" ] && kill "$APP_PID" 2>/dev/null
  [ -n "$STUB_PID" ] && kill "$STUB_PID" 2>/dev/null
}
trap cleanup EXIT

cd "$PROJECT_DIR"
export ECOM_LLM_BASE_URL="http://127.0.0.1:$STUB_PORT/v1"
export ECOM_LLM_MODEL=stub
export ECOM_LLM_API_KEY=stub
export ECOM_LLM_REQUEST_TIMEOUT=30

"$PYTHON" scripts/stub_llm.py --port "$STUB_PORT" >/tmp/check-server-stub.log 2>&1 &
STUB_PID=$!
"$PYTHON" -m uvicorn shopping_assistant.api.app:app --host 127.0.0.1 --port "$PORT" \
  >/tmp/check-server-app.log 2>&1 &
APP_PID=$!

ready=0
for _ in $(seq 1 60); do
  if curl -sS -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  fail "server did not start"
  # Print the logs: on a CI runner this is the only copy.
  echo "--- stub ---"; tail -n 20 /tmp/check-server-stub.log 2>/dev/null
  echo "--- app ---";  tail -n 30 /tmp/check-server-app.log 2>/dev/null
  exit 1
fi
pass "server up on :$PORT"

health=$(curl -sS "http://127.0.0.1:$PORT/health")
echo "$health" | grep -q '"status":"healthy"' && pass "health" || fail "health: $health"

meta=$(curl -sS "http://127.0.0.1:$PORT/api/v1/meta")
echo "$meta" | grep -q '"currency"' && pass "meta: $meta" || fail "meta: $meta"

shape=$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/")
[ "$shape" = "200" ] && pass "frontend shell served" || fail "frontend shell returned $shape"

user=$("$PYTHON" -c "
import json, urllib.request
users = json.load(urllib.request.urlopen('http://127.0.0.1:$PORT/api/v1/users'))
print(users[0]['user_id'])
")
profile=$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/api/v1/users/$user/profile")
[ "$profile" = "200" ] && pass "profile for $user" || fail "profile for $user returned $profile"

stream=$(curl -sS -N -m 120 -X POST "http://127.0.0.1:$PORT/api/v1/chat" \
  -H 'content-type: application/json' \
  -d "{\"user_id\":\"$user\",\"message\":\"推荐一个300元以内的香薰\",\"language\":\"zh\"}")
events=$(echo "$stream" | grep '^event:' | sed 's/^event: //' | tr '\n' ' ')

for expected in session experiment tool products token done; do
  case " $events " in
    *" $expected "*) pass "sse event: $expected" ;;
    *) fail "sse event missing: $expected (got: $events)" ;;
  esac
done

echo "$stream" | grep -q '"product_id"' && pass "product cards carried real catalogue ids" || fail "no product cards in the stream"
echo "$stream" | grep -q '"error"' && fail "the stream reported an error" || pass "no error events"

echo
if [ "$FAILED" -eq 0 ]; then
  echo "SERVER OK: HTTP surface and SSE conversation both work."
else
  echo "SERVER FAILED: $FAILED check(s)."
fi
exit "$FAILED"
