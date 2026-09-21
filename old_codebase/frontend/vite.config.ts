import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// When building for Electron, set VITE_ELECTRON=true env variable.
// This changes the base path to relative so the app loads from file:// in Electron.
const isElectron = process.env.VITE_ELECTRON === "true";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  // Always use relative base path so assets load correctly in both
  // Electron (file://) and the Vite dev server.
  base: "./",
  server: {
    host: "::",
    port: 8080,
    hmr: {
      overlay: false,
    },
    proxy: {
      '/auth': 'http://127.0.0.1:8000',
      '/voice': 'http://127.0.0.1:8000',
      '/audio': 'http://127.0.0.1:8000',
      '/history': 'http://127.0.0.1:8000',
      '/settings': 'http://127.0.0.1:8000',
      '/chat': 'http://127.0.0.1:8000',
      '/api': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/files': 'http://127.0.0.1:8000',
      '/pdf': 'http://127.0.0.1:8000',
      '/mom': 'http://127.0.0.1:8000',
      '/dictionary': 'http://127.0.0.1:8000',
      '/prompt': 'http://127.0.0.1:8000',
    },
  },
  define: {
    // Always point to local backend — works for both Electron (file://) and dev builds
    __ELECTRON_MODE__: JSON.stringify(true),
    __API_BASE_URL__: JSON.stringify('http://127.0.0.1:8000'),
  },
  plugins: [
    react(),
    // Only use lovable-tagger in development (not Electron production builds)
    mode === "development" && !isElectron && componentTagger(),
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
    dedupe: ["react", "react-dom", "react/jsx-runtime", "react/jsx-dev-runtime", "@tanstack/react-query", "@tanstack/query-core"],
  },
}));
