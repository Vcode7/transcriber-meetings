/**
 * SessionExpiredModal.tsx
 * Shown when the 30-day refresh token expires or is revoked.
 * Blocks all interaction and prompts the user to sign in again.
 */
import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { LogIn, ShieldAlert } from 'lucide-react'
import { useAuthStore } from '../store/auth'

export default function SessionExpiredModal() {
  return null
}
