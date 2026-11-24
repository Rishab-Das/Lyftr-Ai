# syntax=docker/dockerfile:1

# ---- build stage: compile the React app to static files ------------------------------
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
# Optional build-time API URL. The container also reads API_URL at start-up (see below).
ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# ---- runtime stage: nginx serving static files as a non-root user --------------------
FROM nginxinc/nginx-unprivileged:1.27-alpine AS runtime
COPY --chown=101:101 nginx.conf /etc/nginx/conf.d/default.conf
COPY --chown=101:101 --chmod=755 docker-entrypoint.sh /docker-entrypoint.d/40-runtime-config.sh
COPY --chown=101:101 --from=build /app/dist /usr/share/nginx/html

# Where the browser finds the backend. Written into /config.js each time the container starts.
ENV API_URL=http://localhost:8000
EXPOSE 3000
