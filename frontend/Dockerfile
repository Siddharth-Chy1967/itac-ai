# ITAC AI frontend — build the Vite/React SPA, serve it with nginx, and proxy
# /api to the backend container so the browser sees a single same-origin app
# (no CORS, and SSE chat streaming passes straight through — see nginx.conf).

# --- build stage ---
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# --- serve stage ---
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
