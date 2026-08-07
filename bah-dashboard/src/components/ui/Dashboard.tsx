import React, { useState, useCallback } from "react";
import type { ChangeEvent } from "react";
import { useNavigate } from "react-router-dom";
import GlassSurface from "./GlassSurface";
import SatelliteScene from "./SatelliteScene";
import SatelliteDishIcon from "./SatelliteDishIcon";

// ─────────────────────────────────────────────────────────────
// Design Tokens
// ─────────────────────────────────────────────────────────────
const original_C = {
  bg:        "#0A0A0B",
  cardBg:    "rgba(20, 20, 22, 0.6)", 
  border:    "rgba(255, 255, 255, 0.08)",
  text:      "#F5F5F5",
  muted:     "#9A9A9E",
  dim:       "#55555A",
  orange:    "#FF6B35",
  blue:      "#4287f5",
  cyan:      "#00D4FF",
  green:     "#3EE07B",
  amber:     "#FFB800",
  red:       "#FF4F5E",
  radius:    "18px",
  radiusSm:  "12px",
};

// ─────────────────────────────────────────────────────────────
// Types & Data
// ─────────────────────────────────────────────────────────────
type FileKey = "tirA" | "wvA" | "tirB" | "wvB" | "tirC" | "wvC";

const FRAME_SLOTS: { key: FileKey; label: string; time: string; required: boolean }[] = [
  { key: "tirA", label: "TIR Frame t0", time: "(Start)", required: true },
  { key: "wvA",  label: "WV Frame t0",  time: "(Start)", required: true },
  { key: "tirB", label: "TIR Frame t1", time: "(End)", required: true },
  { key: "wvB",  label: "WV Frame t1",  time: "(End)", required: true },
  { key: "tirC", label: "TIR Ground Truth", time: "(Midpoint)", required: false },
  { key: "wvC",  label: "WV Ground Truth",  time: "(Midpoint)", required: false },
];

function getStageLabel(progress: number, running: boolean, done: boolean): string {
  if (done) return "DONE";
  if (!running) return "IDLE";
  if (progress < 25) return "FETCHING";
  if (progress < 60) return "INTERPOLATING";
  if (progress < 90) return "SYNTHESISING";
  return "VALIDATING";
}

// ─────────────────────────────────────────────────────────────
// Top Status Bar
// ─────────────────────────────────────────────────────────────
export function StatusBar({ running, done, progress }: { running: boolean; done: boolean; progress: number }) {
  return (
    <GlassSurface width="100%" height={84} borderRadius={0} backgroundOpacity={0.08} blur={14} brightness={40} opacity={0.6} saturation={1.2} style={{ borderBottom: `1px solid rgba(255, 255, 255, 0.06)`, zIndex: 20, flexShrink: 0 }}>
      <div style={{
        height: "100%", width: "100%",
        display: "flex", alignItems: "center",
        padding: "0 2rem",
        gap: "1.5rem",
      }}>
        {/* Animated Tech Logo */}
        <div style={{ 
          width: 44, height: 44, 
          borderRadius: 12, 
          background: `linear-gradient(135deg, rgba(164, 173, 181, 0.15), rgba(164, 173, 181, 0.02))`,
          border: `1px solid rgba(164, 173, 181, 0.3)`,
          display: "flex", alignItems: "center", justifyContent: "center",
          boxShadow: `0 0 16px rgba(164, 173, 181, 0.1)`
        }}>
          <SatelliteDishIcon size={22} color={C.text} strokeWidth={1.5} style={{ filter: `drop-shadow(0 0 6px ${C.muted})` }} />
        </div>

        {/* Title and Badges */}
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <div style={{ 
            fontSize: 20, 
            fontWeight: 700, 
            letterSpacing: "0.05em", 
            background: `linear-gradient(90deg, #FFFFFF, #9A9A9E)`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            fontFamily: "'Aquire', 'Space Grotesk', sans-serif" 
          }}>
            IFNET-GC : Satellite Frame Interpolation
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <div style={{ 
              fontSize: 10, fontWeight: 700, color: C.dim, fontFamily: "'JetBrains Mono', monospace",
              background: "rgba(255,255,255,0.03)", padding: "3px 8px", borderRadius: 4, border: `1px solid rgba(255,255,255,0.06)`
            }}>
              INSAT 3DS, 3DR
            </div>
            <div style={{ 
              fontSize: 10, fontWeight: 700, color: C.dim, fontFamily: "'JetBrains Mono', monospace",
              background: "rgba(255,255,255,0.03)", padding: "3px 8px", borderRadius: 4, border: `1px solid rgba(255,255,255,0.06)`
            }}>
              GOES 16, 19
            </div>
            <div style={{ 
              fontSize: 10, fontWeight: 700, color: "rgba(255,255,255,0.2)", fontFamily: "'JetBrains Mono', monospace",
              background: "transparent", padding: "3px 8px", borderRadius: 4, border: `1px dashed rgba(255,255,255,0.1)`
            }}>
              .nc / .h5
            </div>
          </div>
        </div>
        
        <div style={{ flex: 1 }} />

        {/* Readouts */}
        <div style={{ display: "flex", alignItems: "center", gap: "2rem", marginRight: "1rem" }}>
          {[
            { label: "SYS_LATENCY", value: running ? "24ms" : "4ms", color: C.text },
            { label: "PIPELINE_VER", value: "v2.1.4", color: C.muted },
            { label: "MEMORY_USAGE", value: running ? "14.2 GB" : "1.2 GB", color: running ? C.orange : C.dim }
          ].map(({ label, value, color }) => (
            <div key={label} style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "3px" }}>
              <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: "0.1em", color: C.dim, textTransform: "uppercase" }}>{label}</span>
              <span style={{ fontSize: 13, fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, color }}>{value}</span>
            </div>
          ))}
        </div>

        <div style={{ width: 1, height: 32, background: "rgba(255,255,255,0.08)" }} />

        {/* Live status Pill */}
        <div style={{ 
          display: "flex", alignItems: "center", gap: "10px", 
          background: done ? "rgba(62, 224, 123, 0.08)" : running ? "rgba(255, 107, 53, 0.08)" : "rgba(255,255,255,0.03)",
          border: `1px solid ${done ? "rgba(62, 224, 123, 0.2)" : running ? "rgba(255, 107, 53, 0.2)" : "rgba(255,255,255,0.05)"}`,
          padding: "8px 16px", borderRadius: 999
        }}>
          <div style={{ 
            width: 8, height: 8, borderRadius: "50%", 
            background: done ? C.green : running ? C.orange : C.dim,
            boxShadow: done ? `0 0 10px ${C.green}` : running ? `0 0 10px ${C.orange}` : "none"
          }} />
          <span style={{ 
            fontSize: 12, fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, 
            color: done ? C.green : running ? C.orange : C.dim, 
            textTransform: "uppercase", letterSpacing: "0.05em"
          }}>
            {done ? "COMPLETE" : running ? `PROCESSING ${progress}%` : "STANDBY"}
          </span>
        </div>
      </div>
    </GlassSurface>
  );
}

// ─────────────────────────────────────────────────────────────
// Sidebar Nav
// ─────────────────────────────────────────────────────────────
const NAV_ITEMS = [
  {
    key: "landing",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
        <polyline points="9 22 9 12 15 12 15 22"/>
      </svg>
    ),
    route: "/",
  },
  {
    key: "dashboard",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/>
        <rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>
      </svg>
    ),
    route: "/dashboard",
  },
  {
    key: "architecture",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="12 2 2 7 12 12 22 7 12 2"/>
        <polyline points="2 17 12 22 22 17"/>
        <polyline points="2 12 12 17 22 12"/>
      </svg>
    ),
    route: "/architecture",
  },
];

export function SidebarNav({ active, onNavigate }: { active: string; onNavigate: (route: string, key: string) => void }) {
  const hasResults = Boolean(sessionStorage.getItem("experiment_results"));
  const navItems = [...NAV_ITEMS];
  
  if (hasResults) {
    navItems.push({
      key: "results",
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="16" y1="13" x2="8" y2="13"/>
          <line x1="16" y1="17" x2="8" y2="17"/>
          <polyline points="10 9 9 9 8 9"/>
        </svg>
      ),
      route: "/results",
    });
  }

  return (
    <GlassSurface width={64} height="max-content" borderRadius={999} backgroundOpacity={0.08} blur={14} brightness={40} opacity={0.6} saturation={1.2} style={{ border: `1px solid rgba(255, 255, 255, 0.06)`, borderTop: `1px solid rgba(255, 255, 255, 0.12)`, boxShadow: `0 8px 24px rgba(0,0,0,0.4)`, flexShrink: 0 }}>
      <div style={{
        width: 64, height: "max-content",
        display: "flex", flexDirection: "column", alignItems: "center",
        padding: "1.2rem 0", gap: "0.8rem",
      }}>
      {navItems.map(item => (
        <button key={item.key} onClick={() => onNavigate(item.route, item.key)}
          className="cursor-target"
          style={{
            width: 44, height: 44, borderRadius: "50%",
            border: "1px solid transparent",
            background: active === item.key ? "rgba(255,255,255,0.05)" : "transparent",
            color: active === item.key ? C.orange : C.dim,
            cursor: "pointer",
            display: "flex", alignItems: "center", justifyContent: "center",
            transition: "all 0.2s",
          }}
          title={item.key}
        >{item.icon}</button>
      ))}
      </div>
    </GlassSurface>
  );
}

// ─────────────────────────────────────────────────────────────
// Left Column — Active Runs
// ─────────────────────────────────────────────────────────────
function ActiveRunsList({
  files, progress, running, done, setFiles, runInterpolation,
}: {
  files: Partial<Record<FileKey, File>>;
  progress: number;
  running: boolean;
  done: boolean;
  setFiles: React.Dispatch<React.SetStateAction<Partial<Record<FileKey, File>>>>;
  runInterpolation: () => void;
}) {
  const setFile = (key: FileKey) => (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) setFiles(prev => ({ ...prev, [key]: f }));
  };

  const hasRequiredFiles = !!(files.tirA && files.wvA && files.tirB && files.wvB);

  return (
    <GlassSurface width="100%" height="100%" borderRadius={18} backgroundOpacity={0.08} blur={14} brightness={40} opacity={0.6} saturation={1.2} style={{ border: `1px solid rgba(255, 255, 255, 0.06)`, borderTop: `1px solid rgba(255, 255, 255, 0.12)`, boxShadow: `0 8px 24px rgba(0,0,0,0.4)`, display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "1.5rem 1.5rem 1rem", flexShrink: 0 }}>
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.14em", color: C.muted, textTransform: "uppercase", marginBottom: 6 }}>ACTIVE PROGRAMS</div>
        <div style={{ fontSize: 16, fontWeight: 700, color: C.text }}>Satellite Frames</div>
        <div style={{ fontSize: 13, color: C.dim, marginTop: 4 }}>IFNET-GC Pipeline</div>
      </div>

      {/* Frame slots (Grid) */}
      <div style={{ flex: 1, minHeight: 0, overflowY: "auto", padding: "0 1.5rem 1rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.8rem" }}>
        {FRAME_SLOTS.map(slot => {
          const file = files[slot.key];
          const stageLabel = file ? (done ? "DONE" : running ? getStageLabel(progress, running, done) : "LOADED") : "PENDING";
          const statusColor = file ? (done ? C.green : running ? C.orange : C.dim) : C.dim;

          return (
            <div key={slot.key}
              className="cursor-target"
              style={{
                borderRadius: 12,
                border: file ? "1px solid rgba(255, 255, 255, 0.2)" : "1px solid rgba(255,255,255,0.08)",
                display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "0.4rem",
                padding: "0.6rem 0.5rem",
                cursor: "pointer",
                background: file ? "rgba(255, 255, 255, 0.08)" : "rgba(255,255,255,0.02)",
                transition: "all 0.2s",
                textAlign: "center"
              }}
              onClick={() => {
                const el = document.getElementById(`upload-${slot.key}`);
                if (el) (el as HTMLInputElement).click();
              }}
            >
              <input id={`upload-${slot.key}`} type="file" accept=".nc,.h5" style={{ display: "none" }} onChange={setFile(slot.key)} />
              
              <div style={{ width: 26, height: 26, borderRadius: 6, background: "rgba(255,255,255,0.05)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={file ? C.text : C.muted} strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              </div>

              <div style={{ width: "100%" }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: file ? C.text : C.muted, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", marginBottom: 2 }}>
                  {file ? file.name.slice(0, 16) : slot.label}
                </div>
                <div style={{ fontSize: 11, color: file ? "rgba(255,255,255,0.6)" : C.dim, fontFamily: "'JetBrains Mono', monospace" }}>
                  {file && (running || done) ? `${progress}% - ${stageLabel}` : slot.time}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Actions (Grid) */}
      <div style={{ padding: "0 1.5rem 1.5rem", display: "grid", gridTemplateColumns: "1fr", gap: "1rem", flexShrink: 0 }}>
        <button
          className="cursor-target"
          onClick={runInterpolation}
          disabled={!hasRequiredFiles}
          style={{
            background: !hasRequiredFiles ? "rgba(255,255,255,0.05)" : C.orange,
            color: !hasRequiredFiles ? C.dim : "#0A0A0B",
            border: "none", borderRadius: C.radiusSm, padding: "1rem",
            fontFamily: "'Space Grotesk', sans-serif", fontSize: 13, fontWeight: 600,
            cursor: !hasRequiredFiles ? "not-allowed" : "pointer",
          }}
        >
          Run Interpolation
        </button>
      </div>
    </GlassSurface>
  );
}

import JSZip from "jszip";
// Note: We'll make sure JSZip is imported at the top if it isn't.

function ZipUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [datasetInfo, setDatasetInfo] = useState<any>(null);
  
  const [targetTime, setTargetTime] = useState("");
  const [targetInfo, setTargetInfo] = useState<any>(null);
  const [isTargetLoading, setIsTargetLoading] = useState(false);

  const inputRef = React.useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (!selected) return;
    setFile(selected);
    setIsUploading(true);
    setDatasetInfo(null);
    setTargetInfo(null);
    setTargetTime("");
    
    try {
      const formData = new FormData();
      formData.append("dataset", selected);
      
      const res = await fetch("http://127.0.0.1:8000/experiments/preview-dataset", {
        method: "POST",
        body: formData
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setDatasetInfo(data);
    } catch (e) {
      console.error(e);
      alert("Failed to read dataset: " + e);
      setFile(null);
    } finally {
      setIsUploading(false);
    }
  };

  const handleTargetChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const time = e.target.value;
    setTargetTime(time);
    if (!time) {
      setTargetInfo(null);
      return;
    }
    
    setIsTargetLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/experiments/target", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_time: time,
          frames: datasetInfo.frames
        })
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setTargetInfo(data);
    } catch (e) {
      console.error(e);
      alert("Failed to set target: " + e);
      setTargetInfo(null);
      setTargetTime("");
    } finally {
      setIsTargetLoading(false);
    }
  };

  const handleRun = async () => {
    if (!file || !targetInfo || targetInfo.error) return;
    
    // Check checklist
    const prevOk = targetInfo.previous_channels?.tir && targetInfo.previous_channels?.wv;
    const nextOk = targetInfo.next_channels?.tir && targetInfo.next_channels?.wv;
    if (!prevOk || !nextOk) return;

    try {
      const zip = new JSZip();
      const unzipped = await zip.loadAsync(file);

      const getBlob = async (filename: string) => {
          if (!filename) return null;
          const zf = unzipped.file(filename);
          if (!zf) return null;
          const blob = await zf.async("blob");
          return new File([blob], filename, { type: "application/octet-stream" });
      };

      const tirA = await getBlob(targetInfo.previous_files.tir);
      const wvA = await getBlob(targetInfo.previous_files.wv);
      const tirB = await getBlob(targetInfo.next_files.tir);
      const wvB = await getBlob(targetInfo.next_files.wv);
      
      const tirC = targetInfo.target_files?.tir ? await getBlob(targetInfo.target_files.tir) : null;
      const wvC = targetInfo.target_files?.wv ? await getBlob(targetInfo.target_files.wv) : null;

      navigate("/results", { state: { files: { tirA, wvA, tirB, wvB, tirC, wvC } } });
    } catch (e) {
      console.error("Failed to extract files from zip:", e);
      alert("Failed to read files from ZIP.");
    }
  };
  
  const hasFrames = datasetInfo && datasetInfo.frames;
  const availableTimes = hasFrames ? Object.keys(datasetInfo.frames).sort() : [];

  const prevOk = targetInfo?.previous_channels?.tir && targetInfo?.previous_channels?.wv;
  const nextOk = targetInfo?.next_channels?.tir && targetInfo?.next_channels?.wv;
  const canRun = prevOk && nextOk && !targetInfo?.error;

  return (
    <GlassSurface width="100%" height="auto" borderRadius={18} backgroundOpacity={0.08} blur={14} brightness={40} opacity={0.6} saturation={1.2} style={{ minHeight: "100%", border: `1px solid rgba(255, 255, 255, 0.06)`, borderTop: `1px solid rgba(255, 255, 255, 0.12)`, padding: "1.5rem", display: "flex", flexDirection: "column", gap: "1rem", boxSizing: "border-box" }}>
      <div style={{ fontSize: 16, fontWeight: 700, color: C.text, letterSpacing: "0.05em", fontFamily: "'Space Grotesk', sans-serif" }}>Multiple inputs for optical flow</div>
      
      {!file ? (
        <div className="cursor-target" 
          onClick={() => inputRef.current?.click()}
          style={{ 
          flex: 1, border: `2px dashed ${C.border}`, borderRadius: C.radiusSm, 
          display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "1rem",
          background: "rgba(255,255,255,0.02)", cursor: "pointer", transition: "all 0.2s", minHeight: 200
        }}>
          <input type="file" accept=".zip" ref={inputRef} style={{ display: "none" }} onChange={handleFileChange} />
          <div style={{ width: 40, height: 40, borderRadius: 8, background: "rgba(255,255,255,0.1)", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={C.muted} strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 13, color: C.text, fontWeight: 500 }}>Zip folder containing TIR and WV frames</div>
            <div style={{ fontSize: 12, color: C.dim, marginTop: 4 }}>(Minimum 10 frames)</div>
          </div>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          
          {/* Uploaded File Block */}
          <div style={{ display: "flex", alignItems: "center", gap: "1rem", background: "rgba(255,255,255,0.05)", padding: "1rem", borderRadius: 8, border: `1px solid rgba(255,255,255,0.1)` }}>
            {isUploading ? (
               <div style={{ width: 24, height: 24, borderRadius: "50%", border: `2px solid rgba(255,255,255,0.2)`, borderTopColor: C.orange, animation: "spin 1s linear infinite" }} />
            ) : (
               <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke={C.green} strokeWidth="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            )}
            <div style={{ flex: 1, overflow: "hidden" }}>
              <div style={{ fontSize: 14, color: C.text, fontWeight: 500, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{file.name}</div>
              <div style={{ fontSize: 12, color: C.dim }}>{(file.size / 1024 / 1024).toFixed(2)} MB {isUploading ? "- Scanning Dataset..." : ""}</div>
            </div>
            {!isUploading && (
              <button onClick={() => { setFile(null); setDatasetInfo(null); setTargetInfo(null); }} className="cursor-target" style={{ background: "transparent", border: "none", color: C.muted, cursor: "pointer" }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
              </button>
            )}
          </div>
          
          {/* Dataset Summary */}
          {datasetInfo && (
            <GlassSurface width="100%" height="auto" borderRadius={12} backgroundOpacity={0.05} blur={10} style={{ padding: "0.8rem", border: `1px solid rgba(255,255,255,0.05)` }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: C.text, marginBottom: "0.6rem", fontFamily: "'Space Grotesk', sans-serif" }}>Dataset Summary</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", fontSize: 12, color: C.dim }}>
                <div>Dataset Name: <span style={{ color: C.text }}>{datasetInfo.dataset_name}</span></div>
                <div>Total Files: <span style={{ color: C.text }}>{datasetInfo.total_files}</span></div>
                <div>NC Files: <span style={{ color: C.text }}>{datasetInfo.nc_files}</span></div>
                <div>H5 Files: <span style={{ color: C.text }}>{datasetInfo.h5_files}</span></div>
                <div style={{ gridColumn: "1 / -1" }}>Valid Temporal Frames: <span style={{ color: C.text }}>{datasetInfo.valid_frames_count}</span></div>
              </div>
            </GlassSurface>
          )}

          {/* Target Time Selector */}
          {datasetInfo && (
            <div>
              <label style={{ display: "block", fontSize: 12, color: C.muted, marginBottom: "0.5rem" }}>Interpolation Target</label>
              <select value={targetTime} onChange={handleTargetChange} style={{ width: "100%", background: "rgba(0,0,0,0.4)", border: `1px solid rgba(255,255,255,0.1)`, borderRadius: 6, color: C.text, padding: "0.8rem", fontSize: 13, fontFamily: "'JetBrains Mono', monospace", boxSizing: "border-box", cursor: "pointer", outline: "none" }}>
                <option value="">Select a target time...</option>
                {availableTimes.map(t => (
                  <option key={t} value={t}>{new Date(t).toLocaleString()} ({t.split('T')[1].replace('Z','')} UTC)</option>
                ))}
              </select>
            </div>
          )}

          {/* Selected Frames */}
          {targetInfo && !targetInfo.error && (
             <GlassSurface width="100%" height="auto" borderRadius={12} backgroundOpacity={0.05} blur={10} style={{ padding: "0.8rem", border: `1px solid rgba(255,255,255,0.05)` }}>
               <div style={{ fontSize: 13, fontWeight: 600, color: C.text, marginBottom: "0.8rem", fontFamily: "'Space Grotesk', sans-serif" }}>Selected Frames</div>
               
               <div style={{ display: "flex", flexDirection: "column", gap: "0.8rem" }}>
                 <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                   <div style={{ fontSize: 12, color: C.dim }}>
                     <div style={{ color: C.text, fontWeight: 500 }}>Previous Frame</div>
                     <div>{targetInfo.previous_timestamp.replace('T', ' ')}</div>
                   </div>
                   <div style={{ display: "flex", gap: "0.5rem" }}>
                     <span style={{ fontSize: 10, padding: "2px 6px", borderRadius: 4, background: targetInfo.previous_channels.tir ? "rgba(46, 204, 113, 0.2)" : "rgba(231, 76, 60, 0.2)", color: targetInfo.previous_channels.tir ? C.green : "#E74C3C", border: `1px solid ${targetInfo.previous_channels.tir ? C.green : "#E74C3C"}` }}>TIR {targetInfo.previous_channels.tir ? "✓" : "✗"}</span>
                     <span style={{ fontSize: 10, padding: "2px 6px", borderRadius: 4, background: targetInfo.previous_channels.wv ? "rgba(46, 204, 113, 0.2)" : "rgba(231, 76, 60, 0.2)", color: targetInfo.previous_channels.wv ? C.green : "#E74C3C", border: `1px solid ${targetInfo.previous_channels.wv ? C.green : "#E74C3C"}` }}>WV {targetInfo.previous_channels.wv ? "✓" : "✗"}</span>
                   </div>
                 </div>

                 <div style={{ height: 1, background: "rgba(255,255,255,0.1)", position: "relative" }}>
                   <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", background: "#1a1a1a", padding: "0 8px", fontSize: 11, color: C.orange }}>Target {targetTime.replace('T', ' ')}</div>
                 </div>

                 <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                   <div style={{ fontSize: 12, color: C.dim }}>
                     <div style={{ color: C.text, fontWeight: 500 }}>Next Frame</div>
                     <div>{targetInfo.next_timestamp.replace('T', ' ')}</div>
                   </div>
                   <div style={{ display: "flex", gap: "0.5rem" }}>
                     <span style={{ fontSize: 10, padding: "2px 6px", borderRadius: 4, background: targetInfo.next_channels.tir ? "rgba(46, 204, 113, 0.2)" : "rgba(231, 76, 60, 0.2)", color: targetInfo.next_channels.tir ? C.green : "#E74C3C", border: `1px solid ${targetInfo.next_channels.tir ? C.green : "#E74C3C"}` }}>TIR {targetInfo.next_channels.tir ? "✓" : "✗"}</span>
                     <span style={{ fontSize: 10, padding: "2px 6px", borderRadius: 4, background: targetInfo.next_channels.wv ? "rgba(46, 204, 113, 0.2)" : "rgba(231, 76, 60, 0.2)", color: targetInfo.next_channels.wv ? C.green : "#E74C3C", border: `1px solid ${targetInfo.next_channels.wv ? C.green : "#E74C3C"}` }}>WV {targetInfo.next_channels.wv ? "✓" : "✗"}</span>
                   </div>
                 </div>
               </div>
             </GlassSurface>
          )}

          {/* Interpolation Status */}
          {targetInfo && !targetInfo.error && (
             <GlassSurface width="100%" height="auto" borderRadius={12} backgroundOpacity={0.05} blur={10} style={{ padding: "0.8rem", border: `1px solid rgba(255,255,255,0.05)` }}>
               <div style={{ fontSize: 13, fontWeight: 600, color: C.text, marginBottom: "0.6rem", fontFamily: "'Space Grotesk', sans-serif" }}>Interpolation Status</div>
               <div style={{ display: "flex", flexDirection: "column", gap: "0.3rem", fontSize: 12, color: C.dim }}>
                 <div style={{ display: "flex", justifyContent: "space-between" }}>Dataset Uploaded <span style={{ color: C.green }}>✓</span></div>
                 <div style={{ display: "flex", justifyContent: "space-between" }}>Target Time Selected <span style={{ color: C.green }}>✓</span></div>
                 <div style={{ display: "flex", justifyContent: "space-between" }}>Previous Frame Located <span style={{ color: C.green }}>✓</span></div>
                 <div style={{ display: "flex", justifyContent: "space-between" }}>Next Frame Located <span style={{ color: C.green }}>✓</span></div>
                 <div style={{ display: "flex", justifyContent: "space-between" }}>Previous TIR {targetInfo.previous_channels.tir ? <span style={{ color: C.green }}>✓</span> : <span style={{ color: "#E74C3C" }}>✗</span>}</div>
                 <div style={{ display: "flex", justifyContent: "space-between" }}>Previous WV {targetInfo.previous_channels.wv ? <span style={{ color: C.green }}>✓</span> : <span style={{ color: "#E74C3C" }}>✗</span>}</div>
                 <div style={{ display: "flex", justifyContent: "space-between" }}>Next TIR {targetInfo.next_channels.tir ? <span style={{ color: C.green }}>✓</span> : <span style={{ color: "#E74C3C" }}>✗</span>}</div>
                 <div style={{ display: "flex", justifyContent: "space-between" }}>Next WV {targetInfo.next_channels.wv ? <span style={{ color: C.green }}>✓</span> : <span style={{ color: "#E74C3C" }}>✗</span>}</div>
                 <div style={{ display: "flex", justifyContent: "space-between", marginTop: "0.5rem", paddingTop: "0.5rem", borderTop: "1px solid rgba(255,255,255,0.1)", fontWeight: 600, color: canRun ? C.green : "#E74C3C" }}>
                   Ready to Run <span>{canRun ? "✓" : "✗"}</span>
                 </div>
               </div>
             </GlassSurface>
          )}

          {targetInfo && targetInfo.error && (
            <div style={{ padding: "0.8rem", background: "rgba(231, 76, 60, 0.1)", border: "1px solid rgba(231, 76, 60, 0.3)", borderRadius: 8, color: "#E74C3C", fontSize: 12 }}>
              {targetInfo.error}
            </div>
          )}

          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <button 
              className="cursor-target" 
              disabled={!canRun} 
              onClick={handleRun} 
              title={!canRun && targetInfo ? "Interpolation requires both TIR and WV files for the selected previous and next timestamps." : ""}
              style={{ 
                padding: "0.8rem 1.5rem", 
                background: canRun ? C.orange : "rgba(255,255,255,0.05)", 
                border: "none", color: canRun ? "#000" : C.dim, 
                fontWeight: 700, borderRadius: 8, 
                cursor: canRun ? "pointer" : "not-allowed", 
                transition: "all 0.2s" 
              }}>
                Run Interpolation
            </button>
          </div>
        </div>
      )}
    </GlassSurface>
  );
}

// ─────────────────────────────────────────────────────────────
// Right Column Components
// ─────────────────────────────────────────────────────────────
function DetailsPanel() {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <GlassSurface width="100%" height="100%" borderRadius={18} backgroundOpacity={0.08} blur={14} brightness={40} opacity={0.6} saturation={1.2} style={{ border: `1px solid rgba(255, 255, 255, 0.06)`, borderTop: `1px solid rgba(255, 255, 255, 0.12)`, boxShadow: `0 8px 24px rgba(0,0,0,0.4)` }}>
      <div style={{ height: "100%", width: "100%", display: "flex", flexDirection: "column", overflow: "hidden", position: "relative" }}>
        
        {/* Header */}
        <div style={{ padding: "1.2rem 1.5rem 1rem", flexShrink: 0, position: "relative", zIndex: 10 }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: C.text }}>Details</div>
        </div>
        
        {/* Main 3D Scene with 2D Drawing PIP */}
        <div style={{ flex: 1, minHeight: 0, position: "relative", padding: "0 1.2rem 1.2rem" }}>
          <div style={{ borderRadius: C.radiusSm, overflow: "hidden", height: "100%", background: "rgba(0,0,0,0.5)", border: `1px solid rgba(255,255,255,0.06)`, position: "relative" }}>
            <SatelliteScene done={true} />
            
            {/* Picture-in-Picture 2D Drawing */}
            <div 
              onMouseEnter={() => setIsHovered(true)}
              onMouseLeave={() => setIsHovered(false)}
              style={{ 
              position: "absolute", 
              bottom: "1rem", 
              right: "1rem", 
              width: isHovered ? "calc(100% - 2rem)" : "35%", 
              height: isHovered ? "calc(100% - 2rem)" : "35%", 
              minWidth: isHovered ? "unset" : 180,
              minHeight: isHovered ? "unset" : 140,
              background: "rgba(20, 20, 22, 0.6)", 
              backdropFilter: "blur(12px)", 
              WebkitBackdropFilter: "blur(12px)",
              border: "none", 
              borderRadius: C.radiusSm,
              padding: "1rem",
              boxShadow: isHovered ? "0 8px 32px rgba(0,0,0,0.8)" : "0 4px 24px rgba(0,0,0,0.5)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              overflow: "hidden",
              zIndex: 10,
              transition: "all 0.4s cubic-bezier(0.2, 0.8, 0.2, 1)",
              cursor: isHovered ? "default" : "pointer"
            }}>
              
              {/* Corner Focus Brackets */}
              <div style={{ position: "absolute", top: "1.5rem", left: "1.5rem", width: 24, height: 24, borderTop: "3px solid rgba(255,255,255,0.6)", borderLeft: "3px solid rgba(255,255,255,0.6)", opacity: isHovered ? 1 : 0, transition: "opacity 0.4s 0.2s" }} />
              <div style={{ position: "absolute", top: "1.5rem", right: "1.5rem", width: 24, height: 24, borderTop: "3px solid rgba(255,255,255,0.6)", borderRight: "3px solid rgba(255,255,255,0.6)", opacity: isHovered ? 1 : 0, transition: "opacity 0.4s 0.2s" }} />
              <div style={{ position: "absolute", bottom: "1.5rem", left: "1.5rem", width: 24, height: 24, borderBottom: "3px solid rgba(255,255,255,0.6)", borderLeft: "3px solid rgba(255,255,255,0.6)", opacity: isHovered ? 1 : 0, transition: "opacity 0.4s 0.2s" }} />
              <div style={{ position: "absolute", bottom: "1.5rem", right: "1.5rem", width: 24, height: 24, borderBottom: "3px solid rgba(255,255,255,0.6)", borderRight: "3px solid rgba(255,255,255,0.6)", opacity: isHovered ? 1 : 0, transition: "opacity 0.4s 0.2s" }} />

              <img src="/ref.png" alt="2D Satellite drawing" style={{ width: "100%", height: "100%", objectFit: "contain", transition: "transform 0.4s", transform: isHovered ? "scale(1.02)" : "scale(1)" }} />
            </div>
            
          </div>
        </div>
      </div>
    </GlassSurface>
  );
}

// ─────────────────────────────────────────────────────────────
// Main Dashboard
// ─────────────────────────────────────────────────────────────
function LatLonModal({ isOpen, onClose, onSubmit }: { isOpen: boolean, onClose: () => void, onSubmit: () => void }) {
  if (!isOpen) return null;
  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.8)", backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <GlassSurface width={420} height="auto" borderRadius={18} backgroundOpacity={0.15} blur={20} style={{ padding: "2rem", border: `1px solid rgba(255,255,255,0.1)` }}>
        <h3 style={{ margin: "0 0 0.5rem", color: C.text, fontSize: 18, fontFamily: "'Space Grotesk', sans-serif" }}>Target Coordinates</h3>
        <p style={{ color: C.dim, fontSize: 13, marginBottom: "1.5rem" }}>Enter 4 Latitude/Longitude pairs for the interpolation region.</p>
        
        <div style={{ display: "flex", flexDirection: "column", gap: "0.8rem" }}>
          {[1, 2, 3, 4].map(i => (
            <div key={i} style={{ display: "flex", gap: "1rem", width: "100%" }}>
              <input type="text" placeholder={`Lat ${i}`} style={{ flex: 1, minWidth: 0, background: "rgba(0,0,0,0.4)", border: `1px solid rgba(255,255,255,0.1)`, borderRadius: 6, color: C.text, padding: "0.7rem", fontSize: 13, fontFamily: "'JetBrains Mono', monospace", boxSizing: "border-box" }} />
              <input type="text" placeholder={`Lon ${i}`} style={{ flex: 1, minWidth: 0, background: "rgba(0,0,0,0.4)", border: `1px solid rgba(255,255,255,0.1)`, borderRadius: 6, color: C.text, padding: "0.7rem", fontSize: 13, fontFamily: "'JetBrains Mono', monospace", boxSizing: "border-box" }} />
            </div>
          ))}
        </div>

        <div style={{ display: "flex", gap: "1rem", marginTop: "2rem" }}>
          <button className="cursor-target" onClick={onClose} style={{ flex: 1, padding: "0.8rem", background: "transparent", border: `1px solid rgba(255,255,255,0.1)`, color: C.text, borderRadius: 8, cursor: "pointer", fontWeight: 600 }}>Cancel</button>
          <button className="cursor-target" onClick={onSubmit} style={{ flex: 1, padding: "0.8rem", background: "#FF6B35", border: "none", color: "#000", fontWeight: 700, borderRadius: 8, cursor: "pointer" }}>Confirm & Run</button>
        </div>
      </GlassSurface>
    </div>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [files, setFiles] = useState<Partial<Record<FileKey, File>>>({});
  const [activeNav, setActiveNav] = useState("dashboard");
  const [showLatLonModal, setShowLatLonModal] = useState(false);

  const progress = 0;
  const running = false;
  const done = false;

  const runInterpolation = useCallback(() => {
    navigate("/results", { state: { files } });
  }, [navigate, files]);

  const handleLatLonSubmit = useCallback(() => {
    setShowLatLonModal(false);
    navigate("/results", { state: { files } });
  }, [navigate, files]);

  const handleNav = (route: string, key: string) => {
    setActiveNav(key);
    navigate(route);
  };

  return (
    <>
    <LatLonModal 
      isOpen={showLatLonModal} 
      onClose={() => setShowLatLonModal(false)} 
      onSubmit={handleLatLonSubmit} 
    />
    <div style={{
      fontFamily: "'Aquire', 'Space Grotesk', sans-serif",
      background: "transparent",
      height: "100vh",
      overflow: "hidden",
      display: "flex",
      flexDirection: "column",
      boxSizing: "border-box",
    }}>
      <StatusBar running={running} done={done} progress={progress} />

      <div style={{ display: "flex", flex: 1, minHeight: 0, padding: "2rem", gap: "2rem" }}>
        {/* Sidebar Navigation */}
        <SidebarNav active={activeNav} onNavigate={handleNav} />

        {/* Left Column — Zip Upload */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "1.5rem", height: "100%", overflow: "hidden" }}>
          <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
            <ZipUpload />
          </div>
        </div>

        {/* Right Column — Details / 3D Satellite */}
        <div style={{ flex: 1.5, display: "flex", flexDirection: "column", gap: "2rem", height: "100%", overflow: "hidden" }}>
          <div style={{ flex: 1, minHeight: 0 }}>
            <DetailsPanel />
          </div>
        </div>
      </div>
    </div>
    </>
  );
}

// ─────────────────────────────────────────────────────────────
// Universal Tools
// ─────────────────────────────────────────────────────────────

export { GlassSurface };
export const C = {
  bg:        "#0A0A0B",
  cardBg:    "rgba(20, 20, 22, 0.6)", 
  border:    "rgba(255, 255, 255, 0.08)",
  borderHover:"rgba(255, 255, 255, 0.2)",
  text:      "#FFFFFF",
  dim:       "#A4ADB5",
  muted:     "#6B7280",
  cyan:      "#00E5FF",
  cyanDim:   "rgba(0, 229, 255, 0.15)",
  blue:      "#2563EB",
  orange:    "#FF6B35",
  green:     "#10B981",
  red:       "#EF4444",
  radius:    24,
  radiusSm:  12,
};

export function UniversalChannelViewer({ onResults }: { onResults?: (results: {channel_name: string, url: string}[]) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleGenerate = async () => {
    if (!file) {
      setError("Please select a file first.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      
      const res = await fetch("http://127.0.0.1:8000/universal/channel-viewer", {
        method: "POST",
        body: formData
      });
      
      if (!res.ok) {
        throw new Error(await res.text());
      }
      
      const data = await res.json();
      if (onResults) {
        onResults(data.images);
      }
    } catch (err: any) {
      setError(err.message || "Failed to generate RGB images");
    } finally {
      setLoading(false);
    }
  };

  return (
    <GlassSurface width="100%" height="100%" borderRadius={18} backgroundOpacity={0.08} blur={14} style={{ display: "flex", flexDirection: "column", border: `1px solid rgba(255, 255, 255, 0.06)`, borderTop: `1px solid rgba(255, 255, 255, 0.12)`, padding: "1.5rem" }}>
      <div style={{ fontSize: 14, fontWeight: 700, color: C.text, marginBottom: "0.5rem" }}>Universal Channel Viewer</div>
      <div style={{ fontSize: 12, color: C.dim, marginBottom: "1.5rem" }}>Extract raw RGB composites from any INSAT .nc or .h5 file.</div>
      
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "1rem", justifyContent: "center" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          <label style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.05em" }}>Satellite Data File</label>
          <div style={{ position: "relative" }}>
            <input 
              type="file" 
              className="cursor-target"
              accept=".nc,.h5"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              style={{
                position: "absolute", inset: 0, width: "100%", height: "100%", opacity: 0, zIndex: 10, cursor: "pointer"
              }} 
            />
            <div style={{
              background: "rgba(0,0,0,0.3)",
              border: `1px solid rgba(255,255,255,0.1)`,
              borderRadius: C.radiusSm,
              color: C.text,
              padding: "0.8rem",
              fontSize: 12,
              fontFamily: "'JetBrains Mono', monospace",
              width: "100%",
              boxSizing: "border-box",
              display: "flex",
              alignItems: "center",
              gap: "0.8rem"
            }}>
              <div style={{ background: "rgba(255,255,255,0.1)", padding: "0.3rem 0.6rem", borderRadius: 4, fontSize: 11, fontWeight: "bold" }}>Choose File</div>
              <div style={{ color: file ? C.cyan : C.dim, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {file ? file.name : "Select .nc or .h5 file"}
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {error && <div style={{ color: C.red, fontSize: 12, marginTop: "0.5rem" }}>{error}</div>}
      
      <button 
        className="cursor-target"
        onClick={handleGenerate}
        disabled={loading}
        style={{ 
          marginTop: "1.5rem",
          background: "rgba(255,255,255,0.12)", 
          color: C.text, 
          border: "1px solid rgba(255,255,255,0.1)", 
          padding: "0.8rem", 
          borderRadius: C.radiusSm, 
          fontWeight: 700, 
          cursor: loading ? "not-allowed" : "pointer",
          width: "100%",
          boxShadow: `0 4px 12px rgba(0,0,0,0.2)`,
          opacity: loading ? 0.7 : 1
        }}
      >
        {loading ? "GENERATING..." : "GENERATE RGB"}
      </button>
    </GlassSurface>
  );
}

export function UniversalCropMechanism({ onResults }: { onResults?: (results: {channel_name: string, url: string, filename: string}[]) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [patchSize, setPatchSize] = useState("256");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleExtract = async () => {
    if (!file) {
      setError("Please select a file first.");
      return;
    }
    if (!lat || !lon) {
      setError("Please enter latitude and longitude.");
      return;
    }
    setError("");
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("lat", lat);
      formData.append("lon", lon);
      formData.append("patch_size", patchSize);

      const res = await fetch("http://127.0.0.1:8000/universal/crop", {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Failed to extract crop");
      }

      const data = await res.json();
      if (onResults) {
        onResults(data.results);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <GlassSurface width="100%" height="100%" borderRadius={18} backgroundOpacity={0.08} blur={14} style={{ display: "flex", flexDirection: "column", border: `1px solid rgba(255, 255, 255, 0.06)`, borderTop: `1px solid rgba(255, 255, 255, 0.12)`, padding: "1.5rem" }}>
      <div style={{ fontSize: 14, fontWeight: 700, color: C.text, marginBottom: "0.5rem" }}>Universal Crop Mechanism</div>
      <div style={{ fontSize: 12, color: C.dim, marginBottom: "1rem" }}>Extract a local pixel patch using geographic coordinates.</div>
      
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "0.8rem" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
          <label style={{ fontSize: 11, color: C.muted }}>Satellite Data (.nc/.h5)</label>
          <div style={{ position: "relative" }}>
            <input 
              type="file" 
              className="cursor-target" 
              accept=".nc,.h5" 
              onChange={(e) => setFile(e.target.files?.[0] || null)} 
              style={{ position: "absolute", inset: 0, width: "100%", height: "100%", opacity: 0, zIndex: 10, cursor: "pointer" }} 
            />
            <div style={{
              background: "rgba(0,0,0,0.3)", border: `1px solid rgba(255,255,255,0.1)`, borderRadius: 6, color: C.text, padding: "0.6rem", fontSize: 11, fontFamily: "'JetBrains Mono', monospace", width: "100%", boxSizing: "border-box", display: "flex", alignItems: "center", gap: "0.6rem"
            }}>
              <div style={{ background: "rgba(255,255,255,0.1)", padding: "0.2rem 0.5rem", borderRadius: 4, fontWeight: "bold" }}>Choose File</div>
              <div style={{ color: file ? C.cyan : C.dim, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {file ? file.name : "Select file"}
              </div>
            </div>
          </div>
        </div>
        
        <div style={{ display: "flex", gap: "1rem" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem", flex: 1 }}>
            <label style={{ fontSize: 11, color: C.muted }}>Latitude (°)</label>
            <input type="number" className="cursor-target" placeholder="28.6" value={lat} onChange={(e) => setLat(e.target.value)} style={{ background: "rgba(0,0,0,0.3)", border: `1px solid rgba(255,255,255,0.1)`, borderRadius: 6, color: C.text, padding: "0.6rem", fontSize: 12, fontFamily: "'JetBrains Mono', monospace", width: "100%", boxSizing: "border-box" }} />
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem", flex: 1 }}>
            <label style={{ fontSize: 11, color: C.muted }}>Longitude (°)</label>
            <input type="number" className="cursor-target" placeholder="77.2" value={lon} onChange={(e) => setLon(e.target.value)} style={{ background: "rgba(0,0,0,0.3)", border: `1px solid rgba(255,255,255,0.1)`, borderRadius: 6, color: C.text, padding: "0.6rem", fontSize: 12, fontFamily: "'JetBrains Mono', monospace", width: "100%", boxSizing: "border-box" }} />
          </div>
        </div>
        
        <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
          <label style={{ fontSize: 11, color: C.muted }}>Patch Size (px)</label>
          <input type="number" className="cursor-target" placeholder="256" value={patchSize} onChange={(e) => setPatchSize(e.target.value)} style={{ background: "rgba(0,0,0,0.3)", border: `1px solid rgba(255,255,255,0.1)`, borderRadius: 6, color: C.text, padding: "0.6rem", fontSize: 12, fontFamily: "'JetBrains Mono', monospace", width: "100%", boxSizing: "border-box" }} />
        </div>
        
        {error && <div style={{ color: C.red, fontSize: 12, marginTop: "0.5rem" }}>{error}</div>}

        <button 
          className="cursor-target"
          onClick={handleExtract}
          disabled={loading}
          style={{ 
            marginTop: "1rem",
            background: "rgba(255,255,255,0.12)", 
            color: C.text, 
            border: "1px solid rgba(255,255,255,0.1)", 
            padding: "0.8rem", 
            borderRadius: C.radiusSm, 
            fontWeight: 700, 
            cursor: loading ? "not-allowed" : "pointer",
            width: "100%",
            boxShadow: `0 4px 12px rgba(0,0,0,0.2)`,
            opacity: loading ? 0.7 : 1
          }}
        >
          {loading ? "EXTRACTING..." : "EXTRACT CROP"}
        </button>
      </div>
    </GlassSurface>
  );
}