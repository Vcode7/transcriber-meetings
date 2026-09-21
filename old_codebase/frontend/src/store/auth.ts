/**
 * auth.ts — Persistent auth store
 *
 * Architecture:
 *   - accessToken: stored in localStorage — survives tab/window close and app restarts.
 *     JWT is issued with 100-year expiry so it never expires in practice.
 *   - user: also localStorage for instant initial render (no flicker on reload)
 *   - Refresh token: HttpOnly cookie handled entirely by browser/backend
 *   - sessionExpired: kept for API compatibility but never triggered automatically.
 *     The only way to lose a session is explicit logout.
 *
 * No auto-logout. No idle timeout. No token expiry check.
 */
import { create } from 'zustand'

export interface User {
  id: string
  name: string
  email: string
  needs_setup: boolean
  own_profile_id?: string | null
}

interface AuthState {
  /** Hydrated from localStorage on load — persisted across browser restarts */
  user: User | null
  /** Access token — stored in localStorage so it survives tab/window close */
  accessToken: string | null
  /** Kept for API compatibility — never triggered automatically */
  sessionExpired: boolean
  /** True while the initial bootstrap is running */
  bootstrapping: boolean

  // Actions
  setAuth: (user: User, token: string) => void
  setAccessToken: (token: string) => void
  updateUser: (patch: Partial<User>) => void
  logout: () => void
  setSessionExpired: (val: boolean) => void
  setBootstrapping: (val: boolean) => void
}

export const useAuthStore = create<AuthState>((set) => ({
  // Restore user and token from localStorage for zero-flicker initial render
  user: (() => {
    try {
      return JSON.parse(localStorage.getItem('vs_user') || 'null')
    } catch {
      return null
    }
  })(),
  // Token persisted in localStorage — survives restarts; 100-year JWT never expires
  accessToken: (() => {
    try {
      return localStorage.getItem('vs_access_token') || null
    } catch {
      return null
    }
  })(),
  sessionExpired: false,
  bootstrapping: true,       // starts true; AuthBootstrap clears it

  setAuth: (user, token) => {
    // Persist both user info and token for zero-friction restarts
    localStorage.setItem('vs_user', JSON.stringify(user))
    localStorage.setItem('vs_access_token', token)
    set({ user, accessToken: token, sessionExpired: false })
  },

  setAccessToken: (token) => {
    localStorage.setItem('vs_access_token', token)
    set({ accessToken: token })
  },

  updateUser: (patch) =>
    set((state) => {
      const updated = { ...state.user!, ...patch }
      localStorage.setItem('vs_user', JSON.stringify(updated))
      return { user: updated }
    }),

  logout: () => {
    localStorage.removeItem('vs_user')
    localStorage.removeItem('vs_access_token')
    // The refresh cookie is cleared server-side by POST /auth/logout
    set({ user: null, accessToken: null, sessionExpired: false })
  },

  // Never called automatically — kept for API compatibility only
  setSessionExpired: (val) => set({ sessionExpired: val }),
  setBootstrapping: (val) => set({ bootstrapping: val }),
}))
