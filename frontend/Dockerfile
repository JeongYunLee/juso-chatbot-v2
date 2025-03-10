# Step 1: Build the application
FROM node:18 AS builder

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock) first
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application files
COPY . .

# Build the application
RUN npm run build

# Step 2: Run the application
FROM node:18

# Set working directory
WORKDIR /app

# Copy only the necessary files from the build stage
COPY --from=builder /app/.output /app/.output

# Install only the production dependencies
COPY package*.json ./
RUN npm install --production

# Expose the port the app runs on
EXPOSE 3000

# Start the Nuxt application
CMD ["node", ".output/server/index.mjs"]