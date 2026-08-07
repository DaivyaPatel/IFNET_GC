import React from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { StatusBar, SidebarNav } from "../components/ui/Dashboard";
import GlassSurface from "../components/ui/GlassSurface";

const C = {
  bg:       "#0A0A0B",
  border:   "rgba(255, 255, 255, 0.08)",
  text:     "#F5F5F5",
  muted:    "#9A9A9E",
  dim:      "#55555A",
  orange:   "#FF6B35",
  green:    "#3EE07B",
  red:      "#FF4F5E",
  radiusSm: "12px",
};

export default function ProfilePage() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const initial = user?.email?.[0]?.toUpperCase() || "?";
  const joinDate = user?.created_at
    ? new Date(user.created_at).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })
    : "—";

  return (
    <div style={{ fontFamily: "'Space Grotesk', sans-serif", height: "100vh", overflow: "hidden", display: "flex", flexDirection: "column" }}>
      <StatusBar running={false} done={false} progress={0} />
      <div style={{ display: "flex", flex: 1, minHeight: 0, padding: "2rem", gap: "2rem" }}>
        <SidebarNav active="profile" onNavigate={(r) => navigate(r)} />

        {/* Main content */}
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div style={{ width: "100%", maxWidth: 520, display: "flex", flexDirection: "column", gap: "1.5rem" }}>

            {/* Avatar + Identity */}
            <GlassSurface width="100%" borderRadius={20} backgroundOpacity={0.08} blur={14} style={{ padding: "2rem 2.5rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "1.25rem" }}>
                {/* Avatar */}
                <div style={{ width: 60, height: 60, borderRadius: "50%", background: "linear-gradient(135deg, #FF6B35, #FF8C42)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 26, fontWeight: 800, color: "#0A0A0B", border: "2px solid rgba(255,255,255,0.15)", flexShrink: 0 }}>
                  {initial}
                </div>
                <div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: C.text, letterSpacing: "0.02em" }}>{user?.email || "Unknown Agent"}</div>
                  <div style={{ display: "inline-flex", alignItems: "center", gap: 6, marginTop: 6, padding: "3px 10px", borderRadius: 999, background: "rgba(62, 224, 123, 0.08)", border: "1px solid rgba(62, 224, 123, 0.2)" }}>
                    <div style={{ width: 6, height: 6, borderRadius: "50%", background: C.green, boxShadow: `0 0 8px ${C.green}` }} />
                    <span style={{ fontSize: 11, fontWeight: 700, color: C.green, letterSpacing: "0.1em", textTransform: "uppercase" }}>Active Clearance</span>
                  </div>
                </div>
              </div>
            </GlassSurface>

            {/* Info fields */}
            <GlassSurface width="100%" borderRadius={20} backgroundOpacity={0.08} blur={14} style={{ padding: "1.5rem", display: "flex", flexDirection: "column", gap: "0" }}>
              {[
                { label: "USER_ID", value: user?.id || "—" },
                { label: "EMAIL", value: user?.email || "—" },
                { label: "ACCESS_SINCE", value: joinDate },
                { label: "PIPELINE_VER", value: "v2.1.4" },
                { label: "ROLE", value: "ANALYST" },
              ].map(({ label, value }, i, arr) => (
                <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "1rem 0", borderBottom: i < arr.length - 1 ? `1px solid ${C.border}` : "none" }}>
                  <span style={{ fontSize: 11, fontWeight: 700, color: C.dim, letterSpacing: "0.12em", fontFamily: "'JetBrains Mono', monospace" }}>{label}</span>
                  <span style={{ fontSize: 14, color: C.text, fontFamily: "'JetBrains Mono', monospace", maxWidth: 260, textAlign: "right", wordBreak: "break-all" }}>{value}</span>
                </div>
              ))}
            </GlassSurface>

            {/* Logout button */}
            <button
              onClick={() => { logout(); navigate("/auth"); }}
              style={{
                width: "100%", padding: "1rem", borderRadius: C.radiusSm,
                background: "transparent", border: "1px solid rgba(255, 79, 94, 0.3)",
                color: C.red, fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700,
                fontSize: 15, cursor: "pointer", transition: "all 0.2s",
                display: "flex", alignItems: "center", justifyContent: "center", gap: "0.6rem",
              }}
              onMouseEnter={e => { e.currentTarget.style.background = "rgba(255, 79, 94, 0.1)"; e.currentTarget.style.borderColor = "rgba(255, 79, 94, 0.5)"; }}
              onMouseLeave={e => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.borderColor = "rgba(255, 79, 94, 0.3)"; }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
                <polyline points="16 17 21 12 16 7"/>
                <line x1="21" y1="12" x2="9" y2="12"/>
              </svg>
              Terminate Session
            </button>

          </div>
        </div>
      </div>
    </div>
  );
}
