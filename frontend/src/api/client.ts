
import axios, { AxiosRequestConfig, isAxiosError } from 'axios'
import { useAuthStore } from '../store/auth'

// This is a local desktop app — backend always runs on port 8000.
const BASE_URL = 'http://127.0.0.1:8000'

const api = axios.create({
  baseURL: BASE_URL,
  withCredentials: true,  // REQUIRED — sends HttpOnly refresh cookie automatically
})


// ── State for refresh queue ────────────────────────────────────
let isRefreshing = false
let failedQueue: Array<{
  resolve: (token: string) => void
  reject: (err: unknown) => void
}> = []

function processQueue(error: unknown, token: string | null) {
  failedQueue.forEach((p) => {
    if (error) p.reject(error)
    else p.resolve(token!)
  })
  failedQueue = []
}

// ── Request interceptor — attach access token ──────────────────
api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().accessToken
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// ── Response interceptor — silent refresh on 401 ──────────────
api.interceptors.response.use(
  (res) => res,
  async (error) => {
    const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean }

    // Only handle 401s — and only once per request (avoid infinite loops)
    if (error.response?.status !== 401 || originalRequest._retry) {
      return Promise.reject(error)
    }

    // Don't try to refresh if the failing request IS the refresh or login endpoint
    const url = originalRequest.url || ''
    if (url.includes('/auth/refresh') || url.includes('/auth/login') || url.includes('/auth/register')) {
      return Promise.reject(error)
    }

    originalRequest._retry = true

    if (isRefreshing) {
      // Already refreshing — queue this request and wait for new token
      return new Promise((resolve, reject) => {
        failedQueue.push({ resolve, reject })
      }).then((token) => {
        originalRequest.headers = {
          ...originalRequest.headers,
          Authorization: `Bearer ${token}`,
        }
        return api(originalRequest)
      })
    }

    isRefreshing = true

    try {
      // POST /auth/refresh — browser sends HttpOnly cookie automatically
      const res = await axios.post(
        `${BASE_URL}/auth/refresh`,
        {},
        { withCredentials: true },
      )

      const newToken: string = res.data.access_token
      const user = res.data.user

      // Update in-memory token and user
      useAuthStore.getState().setAuth(user, newToken)

      // Retry all queued requests with new token
      processQueue(null, newToken)

      // Retry the original request
      originalRequest.headers = {
        ...originalRequest.headers,
        Authorization: `Bearer ${newToken}`,
      }
      return api(originalRequest)
    } catch (refreshError: unknown) {
      processQueue(refreshError, null)

      const status = isAxiosError(refreshError) ? refreshError.response?.status : undefined

      // Refresh token expired or invalid — full re-authentication required
      if (status === 401 || status === 403) {
        useAuthStore.getState().logout()
        useAuthStore.getState().setSessionExpired(true)
      }

      return Promise.reject(refreshError)
    } finally {
      isRefreshing = false
    }
  },
)

export default api
