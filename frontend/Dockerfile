# Stage 1: Build React application
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files & install dependencies
COPY package*.json ./
RUN npm ci

# Copy source code and build
COPY . .
ENV REACT_APP_BACKEND_URL=""
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine

# Copy custom Nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built static files from Stage 1
COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
