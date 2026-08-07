import { useState, useRef } from "react";

const NAV_ICONS = {
  dashboard: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
      <rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>
    </svg>
  ),
  layers: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="12 2 2 7 12 12 22 7 12 2"/>
      <polyline points="2 17 12 22 22 17"/>
      <polyline points="2 12 12 17 22 12"/>
    </svg>
  ),
};

const METRICS = [
  { key: "SSIM", value: "0.91", unit: "", color: "#2559bd", bg: "#dae2ff", label: "Structural Similarity" },
  { key: "PSNR", value: "32", unit: "dB", color: "#006e2b", bg: "#e8f5e9", label: "Peak Signal-to-Noise" },
  { key: "MSE", value: "0.004", unit: "", color: "#ba1a1a", bg: "#ffdad6", label: "Mean Squared Error" },
  { key: "FSIM", value: "0.88", unit: "", color: "#6F42C1", bg: "#ede7f6", label: "Feature Similarity" },
];

const STEPS = [
  { label: "Preprocessing", done: true },
  { label: "Optical Flow Estimation", done: true },
  { label: "Frame Synthesis", done: true },
  { label: "Post-processing", done: false },
];

function UploadSlot({ label, sublabel, file, onChange }) {
  const ref = useRef();
  return (
    <div
      onClick={() => ref.current.click()}
      style={{
        border: file ? "1.5px solid #86d1fd" : "1.5px dashed #bfc8cf",
        borderRadius: "0.75rem",
        background: file ? "#f0faff" : "#fff",
        padding: "1rem 0.75rem",
        cursor: "pointer",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "0.4rem",
        minHeight: "110px",
        justifyContent: "center",
        transition: "border-color 0.15s, background 0.15s",
      }}
    >
      <input ref={ref} type="file" accept=".nc,.h5" style={{ display: "none" }} onChange={onChange} />
      {file ? (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00658b" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
      ) : (
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#70787f" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="12" y1="12" x2="12" y2="18"/><line x1="9" y1="15" x2="15" y2="15"/>
        </svg>
      )}
      <span style={{ fontSize: "12px", fontWeight: 600, color: "#191c1e", textAlign: "center", lineHeight: 1.3 }}>{label}</span>
      <span style={{ fontSize: "11px", color: "#70787f", letterSpacing: "0.04em", textTransform: "uppercase", fontWeight: 700 }}>
        {file ? file.name.slice(0, 14) + "…" : sublabel}
      </span>
    </div>
  );
}

function MetricCard({ metric }) {
  return (
    <div style={{
      background: "#fff",
      border: "1px solid #e0e3e6",
      borderRadius: "0.75rem",
      padding: "0.875rem 1rem 0.75rem",
      borderBottom: `4px solid ${metric.color}`,
      display: "flex",
      flexDirection: "column",
      gap: "0.25rem",
    }}>
      <span style={{ fontSize: "11px", fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase", color: metric.color }}>{metric.key}</span>
      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "22px", fontWeight: 600, color: "#191c1e", lineHeight: 1.2 }}>
        {metric.value}<span style={{ fontSize: "13px", fontWeight: 400, color: "#70787f", marginLeft: "2px" }}>{metric.unit}</span>
      </span>
      <span style={{ fontSize: "11px", color: "#70787f" }}>{metric.label}</span>
    </div>
  );
}

export default function Dashboard() {
  const [files, setFiles] = useState({});
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [done, setDone] = useState(false);
  const [activeNav, setActiveNav] = useState("dashboard");
  const [speed, setSpeed] = useState(1);

  const setFile = (key) => (e) => {
    const f = e.target.files[0];
    if (f) setFiles((prev) => ({ ...prev, [key]: f }));
  };

  const runInterpolation = () => {
    if (running) return;
    setRunning(true);
    setDone(false);
    setProgress(0);
    let p = 0;
    const iv = setInterval(() => {
      p += Math.random() * 8 + 2;
      if (p >= 100) {
        p = 100;
        clearInterval(iv);
        setDone(true);
        setRunning(false);
      }
      setProgress(Math.min(Math.round(p), 100));
    }, 200);
  };

  const completedSteps = done ? 4 : running ? Math.floor((progress / 100) * 3) : 0;

  return (
    <div style={{
      fontFamily: "'Hanken Grotesk', 'Inter', sans-serif",
      background: "#f7f9fc",
      minHeight: "100vh",
      padding: "1.5rem",
      boxSizing: "border-box",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={{
        background: "#fff",
        border: "1px solid #e0e3e6",
        borderRadius: "1rem",
        padding: "1rem 1.5rem",
        display: "flex",
        alignItems: "center",
        gap: "1rem",
        marginBottom: "1.25rem",
      }}>
        <div style={{ width: 40, height: 40, borderRadius: "0.5rem", background: "#eceef1", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00658b" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/>
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
          </svg>
        </div>
        <div>
          <div style={{ fontSize: "18px", fontWeight: 700, color: "#191c1e" }}>Satellite Frame Interpolation Dashboard</div>
          <div style={{ fontSize: "12px", color: "#70787f", marginTop: "1px" }}>INSAT-3DS / GOES-19 · IFRNet + PySTEPS Pipeline</div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: done ? "#3E9D52" : running ? "#00658b" : "#bfc8cf", transition: "background 0.3s" }} />
          <span style={{ fontSize: "12px", color: "#70787f", fontWeight: 600 }}>
            {done ? "Complete" : running ? "Processing…" : "Idle"}
          </span>
        </div>
      </div>

      <div style={{ display: "flex", gap: "1.25rem", alignItems: "flex-start" }}>

        {/* Sidebar nav */}
        <div style={{
          width: 52,
          background: "#fff",
          border: "1px solid #e0e3e6",
          borderRadius: "1rem",
          padding: "0.75rem 0",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "0.25rem",
          flexShrink: 0,
        }}>
          {Object.entries(NAV_ICONS).map(([key, icon]) => (
            <button
              key={key}
              onClick={() => setActiveNav(key)}
              style={{
                width: 38,
                height: 38,
                borderRadius: "0.5rem",
                border: "none",
                background: activeNav === key ? "#c4e7ff" : "transparent",
                color: activeNav === key ? "#004c6a" : "#70787f",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                borderLeft: activeNav === key ? "3px solid #00658b" : "3px solid transparent",
                transition: "all 0.15s",
              }}
            >
              {icon}
            </button>
          ))}
        </div>

        {/* Upload panel */}
        <div style={{
          width: 260,
          flexShrink: 0,
          background: "#fff",
          border: "1px solid #e0e3e6",
          borderRadius: "1rem",
          padding: "1.25rem",
          display: "flex",
          flexDirection: "column",
          gap: "1rem",
        }}>
          <span style={{ fontSize: "16px", fontWeight: 700, color: "#191c1e" }}>Upload Frames</span>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.625rem" }}>
            <UploadSlot label="TIR Frame A (T = 0 min)" sublabel=".nc, .h5" file={files.tirA} onChange={setFile("tirA")} />
            <UploadSlot label="WV Frame A (T = 0 min)" sublabel=".nc, .h5" file={files.wvA} onChange={setFile("wvA")} />
            <UploadSlot label="TIR Frame B (T = 10 min)" sublabel=".nc, .h5" file={files.tirB} onChange={setFile("tirB")} />
            <UploadSlot label="WV Frame B (T = 10 min)" sublabel=".nc, .h5" file={files.wvB} onChange={setFile("wvB")} />
            <UploadSlot label="TIR Frame C (T = 20 min)" sublabel=".nc, .h5" file={files.tirC} onChange={setFile("tirC")} />
            <UploadSlot label="WV Frame C (T = 20 min)" sublabel=".nc, .h5" file={files.wvC} onChange={setFile("wvC")} />
          </div>

          <button
            onClick={runInterpolation}
            disabled={running}
            style={{
              background: running ? "#bfc8cf" : "#86d1fd",
              color: running ? "#fff" : "#001e2d",
              border: "none",
              borderRadius: "0.75rem",
              padding: "0.75rem",
              fontFamily: "inherit",
              fontSize: "14px",
              fontWeight: 700,
              cursor: running ? "not-allowed" : "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "0.5rem",
              transition: "background 0.2s",
            }}
          >
            {running ? (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
                </svg>
                Running… {progress}%
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
                Run Interpolation
              </>
            )}
          </button>

          <button style={{
            background: "transparent",
            color: done ? "#191c1e" : "#bfc8cf",
            border: `1px solid ${done ? "#c9d1d9" : "#e0e3e6"}`,
            borderRadius: "0.75rem",
            padding: "0.65rem",
            fontFamily: "inherit",
            fontSize: "13px",
            fontWeight: 600,
            cursor: done ? "pointer" : "not-allowed",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "0.5rem",
          }}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Download Generated Frame (.nc)
          </button>
        </div>

        {/* Main area */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "1.25rem", minWidth: 0 }}>

          {/* Temporal sequence */}
          <div style={{
            background: "#fff",
            border: "1px solid #e0e3e6",
            borderRadius: "1rem",
            padding: "1.25rem",
          }}>
            <span style={{ fontSize: "16px", fontWeight: 700, color: "#191c1e", display: "block", marginBottom: "1rem" }}>Temporal Sequence</span>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              {["Frame B (Original)", "Generated Image"].map((label) => (
                <div key={label} style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                  <div style={{
                    background: "#f2f4f7",
                    border: "1px solid #e0e3e6",
                    borderRadius: "0.75rem",
                    height: "180px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}>
                    {done && label === "Generated Image" ? (
                      <div style={{ textAlign: "center" }}>
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#3E9D52" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                          <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/>
                          <polyline points="21 15 16 10 5 21"/>
                        </svg>
                        <div style={{ fontSize: "12px", color: "#3E9D52", fontWeight: 600, marginTop: "0.5rem" }}>Frame synthesized</div>
                      </div>
                    ) : (
                      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#bfc8cf" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/>
                        <polyline points="21 15 16 10 5 21"/>
                      </svg>
                    )}
                  </div>
                  <span style={{ fontSize: "12px", color: "#70787f", textAlign: "center", fontWeight: 600 }}>{label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Bottom row */}
          <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1.2fr 1fr", gap: "1.25rem" }}>

            {/* Looping animation */}
            <div style={{
              background: "#fff",
              border: "1px solid #e0e3e6",
              borderRadius: "1rem",
              padding: "1.25rem",
              display: "flex",
              flexDirection: "column",
              gap: "0.75rem",
            }}>
              <span style={{ fontSize: "14px", fontWeight: 700, color: "#191c1e" }}>Looping Animation</span>
              <div style={{
                background: "#f2f4f7",
                border: "1px solid #e0e3e6",
                borderRadius: "0.75rem",
                height: "110px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexDirection: "column",
                gap: "0.5rem",
              }}>
                {done ? (
                  <>
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#00658b" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <polygon points="5 3 19 12 5 21 5 3"/>
                    </svg>
                    <span style={{ fontSize: "11px", color: "#00658b", fontWeight: 600 }}>Ready to play</span>
                  </>
                ) : (
                  <span style={{ fontSize: "11px", color: "#bfc8cf", fontWeight: 600 }}>No frames loaded</span>
                )}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                <input type="range" min="0" max="100" defaultValue="60" style={{ width: "100%", accentColor: "#00658b" }} />
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <div style={{ display: "flex", gap: "0.75rem" }}>
                    <button style={{ background: "none", border: "none", cursor: "pointer", color: "#40484e", padding: 0, fontSize: "12px", fontWeight: 600, fontFamily: "inherit" }}>▶ Replay</button>
                    <button style={{ background: "none", border: "none", cursor: "pointer", color: "#40484e", padding: 0, fontSize: "12px", fontWeight: 600, fontFamily: "inherit" }}>⟳ Frame</button>
                  </div>
                  <select
                    value={speed}
                    onChange={(e) => setSpeed(e.target.value)}
                    style={{ border: "1px solid #e0e3e6", borderRadius: "0.375rem", fontSize: "12px", padding: "2px 4px", fontFamily: "inherit", color: "#40484e", background: "#fff" }}
                  >
                    <option value="0.5">0.5×</option>
                    <option value="1">1×</option>
                    <option value="2">2×</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Validation metrics */}
            <div style={{
              background: "#fff",
              border: "1px solid #e0e3e6",
              borderRadius: "1rem",
              padding: "1.25rem",
              display: "flex",
              flexDirection: "column",
              gap: "0.75rem",
            }}>
              <span style={{ fontSize: "14px", fontWeight: 700, color: "#191c1e" }}>Validation Metrics</span>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.625rem" }}>
                {METRICS.map((m) => <MetricCard key={m.key} metric={m} />)}
              </div>
            </div>

            {/* Execution status */}
            <div style={{
              background: "#fff",
              border: "1px solid #e0e3e6",
              borderRadius: "1rem",
              padding: "1.25rem",
              display: "flex",
              flexDirection: "column",
              gap: "0.75rem",
            }}>
              <span style={{ fontSize: "14px", fontWeight: 700, color: "#191c1e" }}>Model Execution Status</span>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.625rem" }}>
                {STEPS.map((step, i) => {
                  const isDone = i < completedSteps;
                  const isActive = i === completedSteps && running;
                  return (
                    <div key={step.label} style={{ display: "flex", alignItems: "center", gap: "0.625rem" }}>
                      <div style={{
                        width: 18, height: 18, borderRadius: "50%",
                        background: isDone ? "#e8f5e9" : isActive ? "#c4e7ff" : "#f2f4f7",
                        border: `1.5px solid ${isDone ? "#3E9D52" : isActive ? "#00658b" : "#e0e3e6"}`,
                        display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                        transition: "all 0.3s",
                      }}>
                        {isDone && (
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#3E9D52" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="20 6 9 17 4 12"/>
                          </svg>
                        )}
                      </div>
                      <span style={{ fontSize: "13px", color: isDone ? "#191c1e" : isActive ? "#004c6a" : "#70787f", fontWeight: isDone || isActive ? 600 : 400, transition: "color 0.3s" }}>
                        {step.label}
                      </span>
                    </div>
                  );
                })}
              </div>
              <div style={{ marginTop: "auto", paddingTop: "0.5rem" }}>
                <div style={{ fontSize: "11px", color: "#70787f", marginBottom: "0.375rem", fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase" }}>
                  Overall progress
                </div>
                <div style={{ background: "#eceef1", borderRadius: "9999px", height: "8px", overflow: "hidden" }}>
                  <div style={{
                    width: `${progress}%`,
                    height: "100%",
                    background: "linear-gradient(90deg, #00658b, #86d1fd)",
                    borderRadius: "9999px",
                    transition: "width 0.2s ease",
                  }} />
                </div>
                <div style={{ fontSize: "12px", color: "#40484e", marginTop: "0.375rem", fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>
                  {progress}%
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}
