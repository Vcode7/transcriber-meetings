import { useEffect } from "react";
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
import SessionExpiredModal from "./components/SessionExpiredModal";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import SquiggleFilter from "@/components/sketch/SquiggleFilter";
import SmoothScroll from "@/components/sketch/SmoothScroll";

const BASE_URL = 'http://127.0.0.1:8000';


/**
 * AuthBootstrap — runs once on app mount.
 *
 * Calls POST /auth/refresh to exchange the HttpOnly cookie for a fresh
 * access token. This is the mechanism that persists login across browser
 * restarts without storing tokens in localStorage.
 *
 * Timeline:
 *   - bootstrapping = true  → render skeleton / nothing for protected routes
 *   - refresh OK            → setAuth(user, token), bootstrapping = false → render app
 *   - refresh fails         → logout(), bootstrapping = false → login page
 */
function AuthBootstrap() {
  const { setAuth, logout, setBootstrapping, user } = useAuthStore();

  useEffect(() => {
    const bootstrap = async () => {
      try {
        const res = await axios.post(
          `${BASE_URL}/auth/refresh`,
          {},
          { withCredentials: true },
        );
        setAuth(res.data.user, res.data.access_token);
      } catch {
        // Refresh failed — clear stale user from localStorage
        logout();
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
 * During bootstrap: shows a minimal loading state instead of flashing /login.
 * After bootstrap: redirects to /login if no user.
 */
function RequireAuth({ children }: { children: React.ReactNode }) {
  const user = useAuthStore((s) => s.user);
  const bootstrapping = useAuthStore((s) => s.bootstrapping);

  if (bootstrapping) {
    // Minimal skeleton — prevents flash of /login during refresh
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
            Restoring session…
          </span>
        </div>
      </div>
    );
  }

  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <TooltipProvider>
      <SquiggleFilter />
      <Toaster />
      <Sonner />
      <HashRouter>
        {/* Bootstrap runs inside HashRouter so SessionExpiredModal can navigate */}
        <AuthBootstrap />
        <SessionExpiredModal />

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
    </TooltipProvider>
  );
}
