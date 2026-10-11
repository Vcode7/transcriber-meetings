/**
 * Global API configuration.
 *
 * Resolves the backend base URL dynamically from environment variables:
 * - VITE_API_BASE_URL (e.g. "http://127.0.0.1:8000" or "http://localhost:8000")
 * - VITE_API_PORT (e.g. "8001" or "8000")
 * - Fallback to __API_BASE_URL__ (injected by Vite define)
 * - Default: "http://127.0.0.1:8000"
 *
 * Change the backend host/port in frontend/.env without modifying source files.
 */

function resolveApiBaseUrl(): string {
  const envBaseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim();
  const envPort = (import.meta.env.VITE_API_PORT as string | undefined)?.trim();

  if (envBaseUrl) {
    if (envPort) {
      try {
        const url = new URL(envBaseUrl);
        url.port = envPort;
        return url.origin;
      } catch {
        return envBaseUrl.replace(/:\d+$/, `:${envPort}`);
      }
    }
    return envBaseUrl.replace(/\/+$/, '');
  }

  if (envPort) {
    return `http://127.0.0.1:${envPort}`;
  }

  if (typeof __API_BASE_URL__ !== 'undefined' && __API_BASE_URL__) {
    return __API_BASE_URL__.replace(/\/+$/, '');
  }

  return 'http://127.0.0.1:8000';
}

export const API_BASE_URL: string = resolveApiBaseUrl();
export const BASE_URL: string = API_BASE_URL;

export default {
  API_BASE_URL,
  BASE_URL,
};
