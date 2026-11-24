#!/bin/sh
# Runs on container start (nginx image executes everything in /docker-entrypoint.d/).
# Writes the runtime config so one image works for any backend URL.
set -eu
# Strip quotes and backslashes so the value cannot break out of the JS string.
SAFE_URL=$(printf '%s' "${API_URL:-}" | tr -d '"\\\n\r')
printf 'window.__CONFIG__ = { API_URL: "%s" };\n' "$SAFE_URL" > /usr/share/nginx/html/config.js
