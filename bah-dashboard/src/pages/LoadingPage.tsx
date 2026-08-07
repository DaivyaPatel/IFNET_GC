import React, { useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import GlassSurface from "../components/ui/GlassSurface";

export default function LoadingPage() {
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    // Navigate to results after a 2.5 second delay to show the animation
    const timer = setTimeout(() => {
      navigate("/results", { state: location.state, replace: true });
    }, 2500);
    return () => clearTimeout(timer);
  }, [navigate, location.state]);

  return (
    <div style={{
      width: "100vw", height: "100vh", display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", background: "#0A0A0B",
      fontFamily: "'Aquire', 'Space Grotesk', sans-serif",
      position: "relative", overflow: "hidden"
    }}>
      {/* Background Starfield - static but nice */}
      <div style={{ 
        position: "absolute", inset: 0, opacity: 0.6, pointerEvents: "none", 
        backgroundImage: "radial-gradient(1px 1px at 20px 30px, rgba(255,255,255,0.8), rgba(0,0,0,0)), radial-gradient(1px 1px at 40px 70px, rgba(255,255,255,0.6), rgba(0,0,0,0)), radial-gradient(1.5px 1.5px at 90px 40px, rgba(255,255,255,0.9), rgba(0,0,0,0)), radial-gradient(1px 1px at 150px 100px, rgba(255,255,255,0.5), rgba(0,0,0,0))", 
        backgroundSize: "200px 200px" 
      }} />

      <GlassSurface width={400} height={400} borderRadius={24} backgroundOpacity={0.05} blur={16} brightness={40} opacity={0.6} saturation={1.2} style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", border: "1px solid rgba(255,255,255,0.05)", boxShadow: "0 16px 40px rgba(0,0,0,0.5)" }}>
        
        {/* Orbital System */}
        <div style={{ position: "relative", width: 160, height: 160, marginBottom: "3rem", marginTop: "-1rem" }}>
          
          {/* Earth Center */}
          <div style={{
            position: "absolute", left: "50%", top: "50%", transform: "translate(-50%, -50%)",
            width: 72, height: 72, borderRadius: "50%",
            background: "radial-gradient(circle at 35% 35%, #4287f5, #0a2e6b 70%, #031026 100%)",
            boxShadow: "0 0 30px rgba(66, 135, 245, 0.4), inset -8px -8px 16px rgba(0,0,0,0.6)",
            zIndex: 2
          }}>
            {/* Subtle atmosphere glow */}
            <div style={{ position: "absolute", inset: -4, borderRadius: "50%", border: "2px solid rgba(66, 135, 245, 0.15)", filter: "blur(2px)" }} />
            {/* Tiny landmass hint */}
            <div style={{ position: "absolute", top: "20%", left: "20%", width: "40%", height: "30%", background: "rgba(62, 224, 123, 0.15)", borderRadius: "40% 60% 70% 30%", filter: "blur(3px)" }} />
            <div style={{ position: "absolute", bottom: "15%", right: "15%", width: "35%", height: "40%", background: "rgba(62, 224, 123, 0.1)", borderRadius: "60% 40% 30% 70%", filter: "blur(4px)" }} />
          </div>

          {/* Orbit Path */}
          <div style={{
            position: "absolute", left: 0, top: 0, width: "100%", height: "100%",
            borderRadius: "50%", border: "1px dashed rgba(255,255,255,0.15)",
            zIndex: 1
          }} />

          {/* Satellite */}
          <div style={{
            position: "absolute", left: 0, top: 0, width: "100%", height: "100%",
            animation: "satellite-orbit 3s linear infinite",
            zIndex: 3
          }}>
            <div style={{
              position: "absolute", top: -14, left: "50%", transform: "translateX(-50%)",
              width: 28, height: 28, display: "flex", alignItems: "center", justifyContent: "center",
              background: "rgba(20,20,22,0.9)", borderRadius: "50%", border: "1px solid rgba(255,255,255,0.3)",
              boxShadow: "0 0 12px rgba(255, 107, 53, 0.4), inset 0 0 8px rgba(0,0,0,0.8)"
            }}>
              {/* Satellite Icon */}
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#FF6B35" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="2" y="7" width="20" height="10" rx="2" ry="2"/>
                <line x1="8" y1="2" x2="8" y2="22"/>
                <line x1="16" y1="2" x2="16" y2="22"/>
              </svg>
            </div>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "0.5rem" }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: "#F5F5F5", letterSpacing: "0.15em", display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ width: 8, height: 8, background: "#FF6B35", borderRadius: "50%", animation: "pulse-orb 1.5s ease-in-out infinite" }} />
            INITIALIZING
          </div>
          <div style={{ fontSize: 12, color: "#9A9A9E", fontFamily: "'JetBrains Mono', monospace" }}>Establishing uplink to IFNET-GC...</div>
        </div>
      </GlassSurface>

      <style>
        {`
          @keyframes satellite-orbit {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          @keyframes pulse-orb {
            0% { transform: scale(0.8); opacity: 0.5; box-shadow: 0 0 0 0 rgba(255, 107, 53, 0.4); }
            50% { transform: scale(1.2); opacity: 1; box-shadow: 0 0 8px 4px rgba(255, 107, 53, 0.2); }
            100% { transform: scale(0.8); opacity: 0.5; box-shadow: 0 0 0 0 rgba(255, 107, 53, 0); }
          }
        `}
      </style>
    </div>
  );
}
