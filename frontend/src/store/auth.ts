/**
 * auth.ts — Production auth store
 *
 * Architecture:
 *   - accessToken: in-memory ONLY (never localStorage) — short-lived (15 min)
 *   - user: localStorage for instant initial render (no flicker on reload)
 *   - Refresh token: HttpOnly cookie handled entirely by browser/backend
 *   - sessionExpired: triggers the global session-expired modal
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
  /** Hydrated from localStorage on load — used for instant render before bootstrap */
  user: User | null
  /** In-memory access token — never persisted. Restored via /auth/refresh on page load */
  accessToken: string | null
  /** True when refresh token has expired — triggers session expired modal */
  sessionExpired: boolean
  /** True while the initial /auth/refresh bootstrap is running */
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
  // Restore user from localStorage for zero-flicker initial render
  user: (() => {
    try {
      return JSON.parse(localStorage.getItem('vs_user') || 'null')
    } catch {
      return null
    }
  })(),
  accessToken: null,         // always starts null; restored by AuthBootstrap
  sessionExpired: false,
  bootstrapping: true,       // starts true; AuthBootstrap clears it

  setAuth: (user, token) => {
    // Persist user info for instant render on reload
    localStorage.setItem('vs_user', JSON.stringify(user))
    // Token is NEVER written to localStorage
    set({ user, accessToken: token, sessionExpired: false })
  },

  setAccessToken: (token) => {
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
    // Do NOT remove vs_token (deprecated) — it was removed in this upgrade
    // The refresh cookie is cleared server-side by POST /auth/logout
    set({ user: null, accessToken: null, sessionExpired: false })
  },

  setSessionExpired: (val) => set({ sessionExpired: val }),
  setBootstrapping: (val) => set({ bootstrapping: val }),
}))
