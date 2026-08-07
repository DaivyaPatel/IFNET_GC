import { BrowserRouter, Routes, Route, useLocation, Navigate } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import AuthPage from "@/pages/AuthPage";
import Dashboard from "@/components/ui/Dashboard";
import ResultsPage from "@/pages/ResultsPage";
import ArchitecturePage from "@/pages/ArchitecturePage";
import ProfilePage from "@/pages/ProfilePage";
import TargetCursor from "@/components/ui/TargetCursor";
import LandingPage from "@/pages/landing/LandingPage";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import './App.css'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, isLoading } = useAuth();
  
  if (isLoading) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#0A0A0B', color: '#FFF', fontFamily: 'Space Grotesk, sans-serif' }}>
        Authenticating...
      </div>
    );
  }
  
  if (!user) {
    return <Navigate to="/auth" replace />;
  }
  
  return <>{children}</>;
}

function CursorWrapper() {
  const location = useLocation();
  if (location.pathname === '/architecture' || location.pathname === '/auth') return null;
  return <TargetCursor key={location.pathname} spinDuration={6} hideDefaultCursor={true} parallaxOn={true} />;
}

function App() {
  return (
    <>
      <BrowserRouter>
        <CursorWrapper />
        <ErrorBoundary>
        <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/landing" element={<Navigate to="/" replace />} />
        <Route path="/auth" element={<AuthPage />} />
        {/* Dashboard — full viewport with black background + space globe */}
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
            <div style={{ position: "relative", minHeight: "100vh", overflow: "hidden", background: "#0A0A0B" }}>
              {/* Background Video Layer */}
              <div
                style={{
                  position: "fixed",
                  top: 0,
                  left: 0,
                  width: "100vw",
                  height: "100vh",
                  zIndex: 0,
                }}
              >
                <img
                  src="/milky-way-bg.jpg"
                  alt="Background"
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                    filter: "saturate(0.8) brightness(0.85)",
                  }}
                />
                <div style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0, background: "rgba(0,0,0,0.7)", pointerEvents: "none" }} />

                {/* Earth Video Layer */}
                <div style={{
                  position: "absolute",
                  left: 0,
                  bottom: "-50%", // Adjusted from -60% to move it slightly up
                  width: "100%",
                  zIndex: 2,
                  pointerEvents: "none",
                  display: "flex",
                  justifyContent: "center"
                }}>
                  <video 
                    onTimeUpdate={(e) => {
                      const video = e.target as HTMLVideoElement;
                      if (video && video.duration && video.currentTime >= video.duration - 0.15) {
                        video.currentTime = 0;
                        video.play().catch(() => {});
                      }
                    }}
                    autoPlay 
                    loop 
                    muted 
                    playsInline 
                    style={{
                      width: "100%",
                      height: "auto",
                      maskImage: "linear-gradient(to top, rgba(0, 0, 0, 1) 75%, rgba(0, 0, 0, 0) 100%)",
                      WebkitMaskImage: "linear-gradient(to top, rgba(0, 0, 0, 1) 75%, rgba(0, 0, 0, 0) 100%)",
                      filter: "brightness(0.6)"
                    }}
                  >
                    <source src="/earth-globe-2.mp4" type="video/mp4" />
                  </video>
                </div>
              </div>
              {/* Dashboard overlay */}
              <div style={{ position: "relative", zIndex: 10, height: "100vh", overflow: "hidden" }}>
                <Dashboard />
              </div>
            </div>
            </ProtectedRoute>
          }
        />

        {/* Architecture explainer — full dark page */}
        <Route
          path="/architecture"
          element={
            <div style={{ position: "relative", minHeight: "100vh", overflow: "hidden", background: "#0A0A0B" }}>
              <div
                style={{
                  position: "fixed",
                  top: 0,
                  left: 0,
                  width: "100vw",
                  height: "100vh",
                  zIndex: 0,
                }}
              >
                <img
                  src="/starfield-4.png"
                  alt="Background"
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                    filter: "saturate(0.6) brightness(0.7)",
                  }}
                />
                <div style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0, background: "rgba(0,0,0,0.8)", pointerEvents: "none" }} />
              </div>
              <div style={{ position: "relative", zIndex: 10, height: "100vh", overflow: "auto" }}>
                <ArchitecturePage />
              </div>
            </div>
          }
        />
        
        {/* Results Page */}
        <Route
          path="/results"
          element={
            <ProtectedRoute>
            <div style={{ position: "relative", minHeight: "100vh", overflow: "hidden", background: "#0A0A0B" }}>
              <div
                style={{
                  position: "fixed",
                  top: 0,
                  left: 0,
                  width: "100vw",
                  height: "100vh",
                  zIndex: 0,
                }}
              >
                <img
                  src="/starfield-4.png"
                  alt="Background"
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                    filter: "saturate(0.6) brightness(0.7)",
                  }}
                />
                <div style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0, background: "rgba(0,0,0,0.8)", pointerEvents: "none" }} />
              </div>
              <div style={{ position: "relative", zIndex: 10, height: "100vh", overflow: "hidden" }}>
                <ResultsPage />
              </div>
            </div>
            </ProtectedRoute>
          }
        />
        {/* Profile Page */}
        <Route
          path="/profile"
          element={
            <ProtectedRoute>
            <div style={{ position: "relative", minHeight: "100vh", overflow: "hidden", background: "#0A0A0B" }}>
              <div style={{ position: "fixed", top: 0, left: 0, width: "100vw", height: "100vh", zIndex: 0 }}>
                <img src="/starfield-4.png" alt="Background" style={{ width: "100%", height: "100%", objectFit: "cover", filter: "saturate(0.6) brightness(0.7)" }} />
                <div style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0, background: "rgba(0,0,0,0.8)", pointerEvents: "none" }} />
              </div>
              <div style={{ position: "relative", zIndex: 10, height: "100vh", overflow: "hidden" }}>
                <ProfilePage />
              </div>
            </div>
            </ProtectedRoute>
          }
        />
      </Routes>
      </ErrorBoundary>
    </BrowserRouter>
    </>
  );
}

export default App;