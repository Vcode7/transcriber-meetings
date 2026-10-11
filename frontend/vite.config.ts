import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// When building for Electron, set VITE_ELECTRON=true env variable.
// This changes the base path to relative so the app loads from file:// in Electron.
const isElectron = process.env.VITE_ELECTRON === "true";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  let apiBaseUrl = (env.VITE_API_BASE_URL || process.env.VITE_API_BASE_URL || "").trim();
  const apiPort = (env.VITE_API_PORT || process.env.VITE_API_PORT || "").trim();

  if (!apiBaseUrl) {
    apiBaseUrl = apiPort ? `http://127.0.0.1:${apiPort}` : "http://127.0.0.1:8000";
  } else if (apiPort) {
    try {
      const parsed = new URL(apiBaseUrl);
      parsed.port = apiPort;
      apiBaseUrl = parsed.origin;
    } catch {
      apiBaseUrl = apiBaseUrl.replace(/:\d+$/, `:${apiPort}`);
    }
  }
  apiBaseUrl = apiBaseUrl.replace(/\/+$/, "");

  return {
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
        '/auth': apiBaseUrl,
        '/voice': apiBaseUrl,
        '/audio': apiBaseUrl,
        '/history': apiBaseUrl,
        '/settings': apiBaseUrl,
        '/chat': apiBaseUrl,
        '/api': apiBaseUrl,
        '/health': apiBaseUrl,
        '/files': apiBaseUrl,
        '/pdf': apiBaseUrl,
        '/mom': apiBaseUrl,
        '/dictionary': apiBaseUrl,
        '/prompt': apiBaseUrl,
        '/ai-chat': apiBaseUrl,
        '/collections': apiBaseUrl,
      },
    },
    define: {
      // Always point to local backend — works for both Electron (file://) and dev builds
      __ELECTRON_MODE__: JSON.stringify(true),
      __API_BASE_URL__: JSON.stringify(apiBaseUrl),
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
  };
});
