import { useEffect, useState } from "react";
import { HashRouter, Routes, Route, Navigate } from "react-router-dom";
import axios from "axios";
import { useAuthStore } from "./store/auth";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Setup from "./pages/Setup";
import Dashboard from "./pages/Dashboard";
import Record from "./pages/Record";
import Upload from "./pages/Upload";
import TabAudio from "./pages/TabAudio";
import History from "./pages/History";
import HistoryDetail from "./pages/HistoryDetail";
import MomPage from "./pages/MomPage";
import AddVoice from "./pages/AddVoice";
import Settings from "./pages/Settings";
import Dictionary from "./pages/Dictionary";
import Landing from "./pages/Landing";
import LicenseExpired from "./pages/LicenseExpired";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import SquiggleFilter from "@/components/sketch/SquiggleFilter";
import SmoothScroll from "@/components/sketch/SmoothScroll";
import GlobalJobTracker from "./components/GlobalJobTracker";

const BASE_URL = "http://127.0.0.1:8000";

// ── License expiry date (must match backend/license.py) ───────
const LICENSE_EXPIRY = new Date("2026-07-30T23:59:59");

/**
 * AuthBootstrap — runs once on app mount.
 *
 * Fast-path: if a valid access token is already stored in localStorage, AND
 * a user record is present, we trust the stored state and skip the /auth/refresh
 * network call entirely — the app loads instantly without any round-trip.
 *
 * Slow-path (no token): calls POST /auth/refresh to exchange the HttpOnly
 * cookie for a fresh access token. This handles the case where the user
 * previously logged in on a different tab that closed between visits.
 *
 * Note: A 401/403 from the refresh endpoint does NOT force a logout — the user
 * keeps their stored session and will see their data. Only an explicit "Sign out"
 * action clears the session.
 */
function AuthBootstrap() {
  const { user, accessToken, setAuth, setBootstrapping } = useAuthStore();

  useEffect(() => {
    const bootstrap = async () => {
      // ── Fast-path ────────────────────────────────────────────
      // Token already in localStorage (100-year JWT — never expires in practice).
      // Trust it immediately; no network round-trip needed.
      if (accessToken && user) {
        setBootstrapping(false);
        return;
      }

      // ── Slow-path ────────────────────────────────────────────
      // No token stored — try to exchange the HttpOnly cookie for a fresh token.
      try {
        const res = await axios.post(
          `${BASE_URL}/auth/refresh`,
          {},
          { withCredentials: true }
        );
        setAuth(res.data.user, res.data.access_token);
      } catch (err: unknown) {
        // Do NOT call logout() on failure.
        // Network errors or backend restart should not sign the user out.
        // The user stays on whatever page they were on; the next API call
        // will retry the refresh automatically via the axios interceptor.
        const status = axios.isAxiosError(err) ? err.response?.status : undefined;
        if (status === 503) {
          // Backend returned license-expired — handled by the license gate below.
          // Nothing to do here; the gate will render LicenseExpired.
        }
        // Any other error (network, 5xx, etc.) — keep stored user active.
      } finally {
        setBootstrapping(false);
      }
    };

    bootstrap();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return null;
}

/**
 * RequireAuth — protects routes.
 *
 * During bootstrap: shows a minimal loading skeleton instead of flashing /login.
 * After bootstrap: redirects to /login if no user in state.
 */
function RequireAuth({ children }: { children: React.ReactNode }) {
  const user = useAuthStore((s) => s.user);
  const bootstrapping = useAuthStore((s) => s.bootstrapping);

  if (bootstrapping) {
    return (
      <div
        style={{
          height: "100dvh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "hsl(var(--background))",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "16px",
          }}
        >
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: "10px",
              background:
                "linear-gradient(135deg, hsl(var(--accent) / .15), hsl(var(--accent) / .05))",
              border: "2px solid hsl(var(--accent) / .3)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <div
              className="spin"
              style={{
                width: 18,
                height: 18,
                border: "2px solid hsl(var(--accent) / .3)",
                borderTop: "2px solid hsl(var(--accent))",
                borderRadius: "50%",
              }}
            />
          </div>
          <span
            style={{
              fontSize: ".8rem",
              color: "hsl(var(--pencil))",
              fontFamily: "Inter, sans-serif",
            }}
          >
            Loading…
          </span>
        </div>
      </div>
    );
  }

  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

/**
 * LicenseGate — wraps the entire app.
 *
 * Checks the license expiry date on the client before any API call.
 * If the local date is past July 30 2026, renders the LicenseExpired
 * blocker immediately (no backend contact needed).
 *
 * Additionally, the AuthBootstrap may receive a 503 from the backend with
 * error="license_expired" — `licenseExpired` state is set accordingly.
 */
function LicenseGate({ children }: { children: React.ReactNode }) {
  // Client-side date check — fast, no network required
  const [expired] = useState<boolean>(() => new Date() > LICENSE_EXPIRY);
  const [serverExpired, setServerExpired] = useState(false);

  useEffect(() => {
    // Cross-check with backend /health — reads license_valid field.
    // /health is allowed through the middleware even after expiry.
    axios
      .get(`${BASE_URL}/health`, { withCredentials: true })
      .then((res) => {
        if (res.data?.license_valid === false) {
          setServerExpired(true);
        }
      })
      .catch((err) => {
        // Also catch cases where middleware returns 503 directly
        if (
          axios.isAxiosError(err) &&
          err.response?.data?.error === "license_expired"
        ) {
          setServerExpired(true);
        }
        // Network errors (backend offline) — do not block the app
      });
  }, []);

  if (expired || serverExpired) {
    return <LicenseExpired />;
  }

  return <>{children}</>;
}

export default function App() {
  return (
    <TooltipProvider>
      <SquiggleFilter />
      <Toaster />
      <Sonner />
      <LicenseGate>
        <HashRouter>
          {/* Bootstrap runs inside HashRouter */}
          <AuthBootstrap />
          {/* GlobalJobTracker: null-render daemon — tracks all in-flight jobs */}
          <GlobalJobTracker />

          <Routes>
            {/* Public landing */}
            <Route path="/" element={<Landing />} />

            {/* Auth */}
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />
            <Route
              path="/setup"
              element={
                <RequireAuth>
                  <Setup />
                </RequireAuth>
              }
            />

            {/* Dashboard (smooth-scroll handled here) */}
            <Route
              path="/dashboard"
              element={
                <RequireAuth>
                  <SmoothScroll>
                    <Dashboard />
                  </SmoothScroll>
                </RequireAuth>
              }
            >
              <Route index element={<Record />} />
              <Route path="tab-audio" element={<TabAudio />} />
              <Route path="upload" element={<Upload />} />
              <Route path="history" element={<History />} />
              <Route path="history/:id" element={<HistoryDetail />} />
              <Route path="history/:id/mom" element={<MomPage />} />
              <Route path="add-voice" element={<AddVoice />} />
              <Route path="settings" element={<Settings />} />
              <Route path="dictionary" element={<Dictionary />} />
            </Route>

            {/* Catch-all → landing */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </HashRouter>
      </LicenseGate>
    </TooltipProvider>
  );
}
