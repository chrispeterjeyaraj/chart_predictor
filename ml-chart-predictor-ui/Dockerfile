# Use an official Node runtime as the base image
FROM node:14-alpine

# Set the working directory in the container
WORKDIR /frontend

# Copy package.json and package-lock.json (if available)
COPY package*.json ./

# Install project dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Build the app for production
RUN npm run build

# Install `serve` to serve the production build
RUN npm install -g serve

# Expose port 3000 for the app
EXPOSE 3000

# Serve the app
CMD ["serve", "-s", "build", "-l", "3000"]