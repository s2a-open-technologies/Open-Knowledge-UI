import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Build geht direkt nach app/static (Flask serviert dieses Verzeichnis als
// /static/...). In dev proxyt Vite alle Backend-Routen an Flask auf :5000.
// base ist im Build '/static/' (Asset-URLs für Flask), im Dev-Server '/',
// damit die SPA samt HMR und Client-Routen direkt unter :5173 läuft.
export default defineConfig(({ mode }) => ({
  base: mode === "production" ? "/static/" : "/",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "./src"),
    },
  },
  build: {
    outDir: "../app/static",
    emptyOutDir: false,
    manifest: false,
  },
  server: {
    port: 5173,
    proxy: {
      // Nur Backend-/API-Pfade an Flask proxyen. Clientseitige React-Router-
      // Routen (/, /login, /dashboard, /documents, /vectordb, /summaries …)
      // werden vom Vite-Dev-Server mit HMR ausgeliefert und dürfen hier NICHT
      // auftauchen.
      "/api": "http://localhost:5000",
      "/chat": "http://localhost:5000",
      "/user": "http://localhost:5000",
      "/logout": "http://localhost:5000",
      "/health": "http://localhost:5000",
      "/ws": { target: "ws://localhost:5000", ws: true },
    },
  },
}));
