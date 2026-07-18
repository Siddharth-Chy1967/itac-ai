import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev-only proxy so the frontend (:5173) reaches the FastAPI backend (:8000).
// Vite's proxy streams responses without buffering, so /api/chat SSE tokens
// arrive incrementally rather than all at once.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
