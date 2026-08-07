import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import JSZip from "jszip";
import { saveAs } from "file-saver";
import GlassSurface from "../components/ui/GlassSurface";
import { StatusBar, SidebarNav } from "../components/ui/Dashboard";
import { transferData, setTransferData } from "../store/transfer";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const C = {
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

function SectionTitle({ title }: { title: string }) {
  return (
    <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 16, fontWeight: 700, color: C.text, marginBottom: "1rem", letterSpacing: "0.05em" }}>
      {title}
    </div>
  );
}

function ImagePlaceholder({ label, src }: { label: string; src?: string | null }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1rem", height: "100%" }}>
      <div style={{ 
        flex: 1, width: "100%", background: "rgba(255,255,255,0.03)", 
        border: `1px solid ${C.border}`, borderRadius: C.radiusSm,
        position: "relative", overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center"
      }}>
        {src ? (
          <img src={src} style={{ width: "100%", height: "100%", objectFit: "contain" }} alt={label} />
        ) : (
          <svg width="100%" height="100%" style={{ position: "absolute", inset: 0, opacity: 0.2 }}>
            <line x1="0" y1="0" x2="100%" y2="100%" stroke={C.muted} strokeWidth="1" />
            <line x1="100%" y1="0" x2="0" y2="100%" stroke={C.muted} strokeWidth="1" />
          </svg>
        )}
      </div>
      <div style={{ fontSize: 13, color: C.text, fontFamily: "'JetBrains Mono', monospace", textAlign: "center" }}>
        {label}
      </div>
    </div>
  );
}

function TemporalSequence({
  t0Url, t1Url, midTruthUrl, generatedUrl, gapMinutes
}: {
  t0Url?: string | null; t1Url?: string | null; midTruthUrl?: string | null; generatedUrl?: string | null; gapMinutes?: number | null;
}) {
  return (
    <GlassSurface width="100%" height="100%" borderRadius={18} backgroundOpacity={0.08} blur={14} style={{ padding: "1.5rem", display: "flex", flexDirection: "column" }}>
      <SectionTitle title="Temporal Sequence" />
      <div style={{ flex: 1, display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1.5rem", minHeight: 0 }}>
        <ImagePlaceholder label="Frame t0 (Start)" src={t0Url} />
        <ImagePlaceholder label={`Ground Truth (T=${gapMinutes ? gapMinutes/2 : '?'} min)`} src={midTruthUrl} />
        <ImagePlaceholder label={`Generated (T=${gapMinutes ? gapMinutes/2 : '?'} min)`} src={generatedUrl} />
        <ImagePlaceholder label={`Frame t1 (T=${gapMinutes || '?'} min)`} src={t1Url} />
      </div>
    </GlassSurface>
  );
}

function MotionVectors({ realUrl, interpolatedUrl }: { realUrl?: string | null; interpolatedUrl?: string | null }) {
  return (
    <GlassSurface width="100%" height="100%" borderRadius={18} backgroundOpacity={0.08} blur={14} style={{ padding: "1.5rem", display: "flex", flexDirection: "column" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
        <SectionTitle title="HSV Motion Vectors" />
        <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: C.dim, fontFamily: "'JetBrains Mono', monospace" }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#FF6B35" }} />&nbsp;Farneback Optical Flow
        </div>
      </div>
      <div style={{ flex: 1, display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem", minHeight: 0 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", height: "100%" }}>
          <div style={{ flex: 1, borderRadius: C.radiusSm, overflow: "hidden", border: `1px solid ${C.border}`, background: "#000", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
            {realUrl ? (
              <img src={realUrl} style={{ width: "100%", height: "100%", objectFit: "contain" }} alt="Ground Truth Flow" />
            ) : (
              <span style={{ color: C.dim, fontFamily: "'JetBrains Mono', monospace", fontSize: 12 }}>Awaiting computation...</span>
            )}
          </div>
          <div style={{ fontSize: 12, color: C.muted, textAlign: "center", fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.06em" }}>GROUND TRUTH FLOW</div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", height: "100%" }}>
          <div style={{ flex: 1, borderRadius: C.radiusSm, overflow: "hidden", border: `1px solid ${C.border}`, background: "#000", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
            {interpolatedUrl ? (
              <img src={interpolatedUrl} style={{ width: "100%", height: "100%", objectFit: "contain" }} alt="Predicted Flow" />
            ) : (
              <span style={{ color: C.dim, fontFamily: "'JetBrains Mono', monospace", fontSize: 12 }}>Awaiting computation...</span>
            )}
          </div>
          <div style={{ fontSize: 12, color: C.muted, textAlign: "center", fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.06em" }}>PREDICTED FLOW</div>
        </div>
      </div>
    </GlassSurface>
  );
}

function DifferenceMap({ src, metrics }: { src?: string | null, metrics?: any }) {
  return (
    <GlassSurface width="100%" height="100%" borderRadius={18} backgroundOpacity={0.08} blur={14} style={{ padding: "1.5rem", display: "flex", flexDirection: "column" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
        <SectionTitle title="Real vs Interpolated Intensity" />
        {metrics?.mae != null && (
          <div style={{ 
            fontFamily: "'JetBrains Mono', monospace", 
            fontSize: 14, 
            color: C.text,
            background: "rgba(255,255,255,0.1)",
            padding: "4px 12px",
            borderRadius: 20
          }}>
            % Diff: <span style={{ color: C.red, fontWeight: "bold" }}>{(metrics.mae / 255.0 * 100).toFixed(2)}%</span>
          </div>
        )}
      </div>
      <div style={{ flex: 1, position: "relative", borderRadius: C.radiusSm, overflow: "hidden", border: `1px solid ${C.border}`, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.4)" }}>
        {src ? (
          <img src={src} style={{ width: "100%", height: "100%", objectFit: "contain" }} alt="Correlation Graph" />
        ) : (
          <div style={{ color: C.dim, fontFamily: "'JetBrains Mono', monospace" }}>No Graph Available</div>
        )}
      </div>
    </GlassSurface>
  );
}

function CycloneTrackerMetrics({ tracking }: { tracking?: any }) {
  if (!tracking) return null;
  
  const mList = [
    { label: "SPEED", value: `${tracking.speed_kmh} km/h`, color: C.green },
    { label: "COMPASS", value: tracking.compass, color: C.text },
    { label: "DISPLACEMENT", value: `${tracking.distance_km} km`, color: C.cyan },
    { label: "VECTOR", value: `[${tracking.vx_kmh}, ${tracking.vy_kmh}]`, color: C.text },
    { label: "DIRECTION", value: `${tracking.direction_deg}°`, color: C.orange },
    { label: "CONFIDENCE", value: tracking.confidence, color: C.amber },
  ];

  return (
    <GlassSurface width="100%" height="100%" borderRadius={18} backgroundOpacity={0.08} blur={14} style={{ padding: "1.5rem", display: "flex", flexDirection: "column" }}>
      <SectionTitle title="Cyclone Tracking (TIR)" />
      <div style={{ flex: 1, display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gridTemplateRows: "1fr 1fr", border: `1px solid ${C.border}`, borderRadius: C.radiusSm, overflow: "hidden", marginTop: "1rem" }}>
        {mList.map((m, i) => {
          let gridColumn = "span 2";
          return (
          <div key={m.label} style={{ 
            display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "1rem 0",
            borderRight: (i === 0 || i === 1 || i === 3 || i === 4) ? `1px solid ${C.border}` : 'none',
            borderBottom: i < 3 ? `1px solid ${C.border}` : 'none',
            background: "rgba(255,255,255,0.02)",
            gridColumn
          }}>
            <div style={{ fontSize: 20, fontWeight: 700, color: m.color, fontFamily: "'JetBrains Mono', monospace", textShadow: `0 0 12px ${m.color}66` }}>{m.value}</div>
            <div style={{ fontSize: 11, color: C.dim, marginTop: 4, letterSpacing: "0.05em", textTransform: "uppercase" }}>{m.label}</div>
          </div>
        )})}
      </div>
    </GlassSurface>
  );
}

function IntensityChart({ data }: { data: any }) {
  if (!data) return (
    <GlassSurface width="100%" height="100%" borderRadius={18} backgroundOpacity={0.08} blur={14} style={{ padding: "1.5rem", display: "flex", flexDirection: "column" }}>
      <SectionTitle title="Average Intensity vs Time" />
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <span style={{ color: C.dim }}>No Data Available</span>
      </div>
    </GlassSurface>
  );

  const chartData = data.labels.map((label: string, i: number) => ({
    name: label,
    real: data.real[i],
    interpolated: data.interpolated[i]
  }));

  return (
    <GlassSurface width="100%" height="100%" borderRadius={18} backgroundOpacity={0.08} blur={14} style={{ padding: "1.5rem", display: "flex", flexDirection: "column" }}>
      <SectionTitle title="Average Intensity vs Time" />
      <div style={{ flex: 1, minHeight: 0, marginTop: "1rem" }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="name" stroke={C.dim} tick={{ fill: C.dim }} />
            <YAxis stroke={C.dim} tick={{ fill: C.dim }} domain={['auto', 'auto']} />
            <Tooltip 
              contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: `1px solid ${C.border}`, borderRadius: C.radiusSm }}
              itemStyle={{ fontFamily: "'JetBrains Mono', monospace" }}
            />
            <Legend wrapperStyle={{ paddingTop: "1rem" }} />
            <Line type="monotone" dataKey="real" stroke={C.green} strokeWidth={2} dot={{ r: 5 }} activeDot={{ r: 8 }} name="Real" />
            <Line type="monotone" dataKey="interpolated" stroke={C.orange} strokeWidth={2} strokeDasharray="5 5" dot={{ r: 5 }} activeDot={{ r: 8 }} name="Interpolated" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </GlassSurface>
  );
}

function EyeTrackingChart({ data, metrics }: { data: any, metrics?: any }) {
  if (!data) return (
    <GlassSurface width="100%" height="100%" borderRadius={18} backgroundOpacity={0.08} blur={14} style={{ padding: "1.5rem", display: "flex", flexDirection: "column" }}>
      <SectionTitle title="Cyclone Eye Tracking" />
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <span style={{ color: C.dim }}>No Data Available</span>
      </div>
    </GlassSurface>
  );

  const chartData = data.labels.map((label: string, i: number) => ({
    name: label,
    real_x: data.real_x[i],
    real_y: data.real_y[i],
    interp_x: data.interp_x[i],
    interp_y: data.interp_y[i]
  }));

  return (
    <GlassSurface width="100%" height="100%" borderRadius={18} backgroundOpacity={0.08} blur={14} style={{ padding: "1.5rem", display: "flex", flexDirection: "column" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <SectionTitle title="Cyclone Eye Tracking" />
        {metrics?.expected_vs_detected_km != null && (
          <div style={{ 
            fontFamily: "'JetBrains Mono', monospace", 
            fontSize: 12, 
            color: C.text,
            background: "rgba(255,255,255,0.1)",
            padding: "4px 10px",
            borderRadius: 20,
            marginBottom: "1rem"
          }}>
            Error: <span style={{ color: C.red, fontWeight: "bold" }}>{metrics.expected_vs_detected_km} km</span>
          </div>
        )}
      </div>
      <div style={{ flex: 1, minHeight: 0, marginTop: "0.5rem" }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="name" stroke={C.dim} tick={{ fill: C.dim }} />
            <YAxis yAxisId="x" stroke={C.dim} tick={{ fill: C.dim }} domain={['auto', 'auto']} />
            <YAxis yAxisId="y" orientation="right" reversed stroke={C.dim} tick={{ fill: C.dim }} domain={['auto', 'auto']} />
            <Tooltip 
              contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: `1px solid ${C.border}`, borderRadius: C.radiusSm }}
              itemStyle={{ fontFamily: "'JetBrains Mono', monospace" }}
            />
            <Legend wrapperStyle={{ paddingTop: "1rem" }} />
            <Line yAxisId="x" type="monotone" dataKey="real_x" stroke={C.cyan} strokeWidth={2} dot={{ r: 5 }} activeDot={{ r: 8 }} name="Real X" />
            <Line yAxisId="y" type="monotone" dataKey="real_y" stroke={C.blue} strokeWidth={2} dot={{ r: 5 }} activeDot={{ r: 8 }} name="Real Y" />
            <Line yAxisId="x" type="monotone" dataKey="interp_x" stroke={C.red} strokeWidth={2} strokeDasharray="5 5" dot={{ r: 5 }} activeDot={{ r: 8 }} name="Interp X" />
            <Line yAxisId="y" type="monotone" dataKey="interp_y" stroke={C.amber} strokeWidth={2} strokeDasharray="5 5" dot={{ r: 5 }} activeDot={{ r: 8 }} name="Interp Y" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </GlassSurface>
  );
}

function LoopingAnimation({ 
  title = "Looping Animation", 
  frames,
  currentFrame,
  setCurrentFrame,
  isPlaying,
  setIsPlaying,
  speed,
  setSpeed
}: { 
  title?: string; 
  frames?: (string | null)[];
  currentFrame: number;
  setCurrentFrame: React.Dispatch<React.SetStateAction<number>>;
  isPlaying: boolean;
  setIsPlaying: React.Dispatch<React.SetStateAction<boolean>>;
  speed: number;
  setSpeed: React.Dispatch<React.SetStateAction<number>>;
}) {
  const validFrames = frames ? frames.filter(f => f) as string[] : [];
  const hasFrames = validFrames.length > 0;
  const frameToDisplay = validFrames[currentFrame % Math.max(1, validFrames.length)];

  return (
    <GlassSurface width="100%" height="100%" borderRadius={18} backgroundOpacity={0.08} blur={14} style={{ padding: "1.5rem", display: "flex", flexDirection: "column" }}>
      <SectionTitle title={title} />
      <div style={{ 
        flex: 1, background: "#050505", border: `1px solid ${C.border}`, 
        borderRadius: C.radiusSm, display: "flex", flexDirection: "column", overflow: "hidden"
      }}>
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.4)", position: "relative" }}>
          {hasFrames ? (
            <img src={frameToDisplay} style={{ width: "100%", height: "100%", objectFit: "contain" }} alt={`${title} frame ${currentFrame}`} />
          ) : (
            <>
              <div style={{ position: "absolute", inset: 0, opacity: 0.2, background: "linear-gradient(45deg, #111 25%, transparent 25%, transparent 75%, #111 75%, #111), linear-gradient(45deg, #111 25%, transparent 25%, transparent 75%, #111 75%, #111)", backgroundSize: "20px 20px", backgroundPosition: "0 0, 10px 10px" }} />
              <div className="cursor-target" style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", zIndex: 2 }}>
                <div style={{ color: C.dim, fontFamily: "'JetBrains Mono', monospace" }}>Awaiting frames...</div>
              </div>
            </>
          )}
        </div>
        
        <div style={{ 
          height: 48, background: "rgba(10, 10, 11, 0.8)", borderTop: `1px solid ${C.border}`,
          display: "flex", alignItems: "center", padding: "0 1rem", gap: "1rem"
        }}>
          <button 
            onClick={() => setIsPlaying(!isPlaying)}
            style={{ background: "transparent", border: "none", color: C.text, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", padding: 4 }}
          >
            {isPlaying ? (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            )}
          </button>

          <button 
            onClick={() => {
              setIsPlaying(false);
              setCurrentFrame(prev => (prev - 1 + validFrames.length) % validFrames.length);
            }}
            style={{ background: "transparent", border: "none", color: C.muted, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", padding: 4 }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="19 20 9 12 19 4 19 20"/><line x1="5" y1="19" x2="5" y2="5"/></svg>
          </button>
          
          <button 
            onClick={() => {
              setIsPlaying(false);
              setCurrentFrame(prev => (prev + 1) % validFrames.length);
            }}
            style={{ background: "transparent", border: "none", color: C.muted, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", padding: 4 }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="5 4 15 12 5 20 5 4"/><line x1="19" y1="5" x2="19" y2="19"/></svg>
          </button>

          <div style={{ flex: 1, display: "flex", gap: 4, height: 4 }}>
            {validFrames.map((_, i) => (
              <div 
                key={i} 
                onClick={() => { setIsPlaying(false); setCurrentFrame(i); }}
                style={{ 
                  flex: 1, 
                  background: i === (currentFrame % validFrames.length) ? C.orange : "rgba(255,255,255,0.2)", 
                  borderRadius: 2,
                  cursor: "pointer",
                  transition: "background 0.2s"
                }} 
              />
            ))}
          </div>

          <select 
            value={speed} 
            onChange={e => setSpeed(Number(e.target.value))}
            style={{ 
              background: "transparent", border: "none", color: C.text, fontFamily: "'JetBrains Mono', monospace", 
              fontSize: 12, cursor: "pointer", outline: "none" 
            }}
          >
            <option value={0.5} style={{ background: C.bg }}>0.5x</option>
            <option value={1} style={{ background: C.bg }}>1x</option>
            <option value={2} style={{ background: C.bg }}>2x</option>
            <option value={4} style={{ background: C.bg }}>4x</option>
          </select>
        </div>
      </div>
    </GlassSurface>
  );
}

function ValidationMetrics({ metrics }: { metrics?: any }) {
  const mList = [
    { label: "SSIM", value: metrics?.ssim ? metrics.ssim.toFixed(4) : "N/A", color: C.cyan },
    { label: "PSNR", value: metrics?.psnr ? metrics.psnr.toFixed(2) : "N/A", color: C.orange },
    { label: "RMSE", value: metrics?.rmse ? metrics.rmse.toFixed(4) : "N/A", color: C.green },
    { label: "FSIM", value: metrics?.fsim ? metrics.fsim.toFixed(4) : "N/A", color: C.amber },
    { label: "LPIPS", value: metrics?.lpips ? metrics.lpips.toFixed(4) : "N/A", color: "#E74C3C" },
  ];
  return (
    <GlassSurface width="100%" height="100%" borderRadius={18} backgroundOpacity={0.08} blur={14} style={{ padding: "1.5rem", display: "flex", flexDirection: "column" }}>
      <SectionTitle title="Validation Metrics" />
      <div style={{ flex: 1, display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gridTemplateRows: "1fr 1fr", border: `1px solid ${C.border}`, borderRadius: C.radiusSm, overflow: "hidden" }}>
        {mList.map((m, i) => {
          let gridColumn = "span 2";
          if (i === 3) gridColumn = "2 / span 2";
          if (i === 4) gridColumn = "4 / span 2";

          return (
          <div key={m.label} style={{ 
            display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "1rem 0",
            borderRight: (i === 0 || i === 1 || i === 3) ? `1px solid ${C.border}` : 'none',
            borderBottom: i < 3 ? `1px solid ${C.border}` : 'none',
            background: "rgba(255,255,255,0.02)",
            gridColumn
          }}>
            <div style={{ fontSize: 22, fontWeight: 700, color: m.color, fontFamily: "'JetBrains Mono', monospace", textShadow: `0 0 12px ${m.color}66` }}>{m.value}</div>
            <div style={{ fontSize: 11, color: C.dim, marginTop: 4, letterSpacing: "0.05em", textTransform: "uppercase" }}>{m.label}</div>
          </div>
        )})}
      </div>
    </GlassSurface>
  );
}

function DownloadResultsCard({
  done, t0Url, t1Url, midTruthUrl, generatedUrl, realGifUrl, interpolatedGifUrl, hsvFlowRealUrl, hsvFlowInterpolatedUrl
}: {
  done: boolean;
  t0Url?: string | null; t1Url?: string | null; midTruthUrl?: string | null;
  generatedUrl?: string | null; realGifUrl?: string | null; interpolatedGifUrl?: string | null;
  hsvFlowRealUrl?: string | null; hsvFlowInterpolatedUrl?: string | null;
}) {
  const [isDownloading, setIsDownloading] = useState(false);

  const handleDownload = async () => {
    if (!done || isDownloading) return;
    setIsDownloading(true);
    const zip = new JSZip();
    const fetchToZip = async (url: string, filename: string) => {
      try {
        const response = await fetch(url);
        const blob = await response.blob();
        zip.file(filename, blob);
      } catch (e) {
        console.error("Failed to fetch", url, e);
      }
    };
    
    if (t0Url) await fetchToZip(t0Url, "t0_composite.png");
    if (t1Url) await fetchToZip(t1Url, "t1_composite.png");
    if (midTruthUrl) await fetchToZip(midTruthUrl, "ground_truth_midpoint.png");
    if (generatedUrl) await fetchToZip(generatedUrl, "generated_midpoint.png");
    if (realGifUrl) await fetchToZip(realGifUrl, "real_animation.gif");
    if (interpolatedGifUrl) await fetchToZip(interpolatedGifUrl, "interpolated_animation.gif");
    if (hsvFlowRealUrl) await fetchToZip(hsvFlowRealUrl, "hsv_flow_real.png");
    if (hsvFlowInterpolatedUrl) await fetchToZip(hsvFlowInterpolatedUrl, "hsv_flow_interpolated.png");
    
    const content = await zip.generateAsync({ type: "blob" });
    saveAs(content, "interpolation_results.zip");
    setIsDownloading(false);
  };

  return (
    <GlassSurface width="100%" height="100%" borderRadius={18} backgroundOpacity={0.08} blur={14} style={{ padding: "2rem", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "2rem" }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "0.5rem" }}>
        <SectionTitle title="Export Data" />
        <span style={{ fontSize: 13, color: C.dim, textAlign: "center", maxWidth: "80%" }}>
          Download all generated outputs, animations, and intermediate maps as a ZIP archive.
        </span>
      </div>
      <button
        className="cursor-target"
        onClick={handleDownload}
        disabled={!done || isDownloading}
        onMouseDown={(e) => {
          if (done && !isDownloading) {
            e.currentTarget.style.transform = "scale(0.96)";
            e.currentTarget.style.filter = "brightness(0.9)";
          }
        }}
        onMouseUp={(e) => {
          if (done) {
            e.currentTarget.style.transform = "scale(1)";
            e.currentTarget.style.filter = "brightness(1)";
          }
        }}
        onMouseLeave={(e) => {
          if (done) {
            e.currentTarget.style.transform = "scale(1)";
            e.currentTarget.style.filter = "brightness(1)";
          }
        }}
        style={{
          background: done ? (isDownloading ? C.dim : C.orange) : "rgba(255,255,255,0.05)",
          color: done ? "#0A0A0B" : C.dim,
          border: "none", borderRadius: C.radiusSm, padding: "1rem 2rem",
          fontFamily: "'Space Grotesk', sans-serif", fontSize: 14, fontWeight: 700,
          cursor: done ? "pointer" : "not-allowed",
          transition: "all 0.2s",
          width: "100%",
          display: "flex", alignItems: "center", justifyContent: "center", gap: "0.5rem"
        }}
      >
        {isDownloading ? (
          "Packaging Zip..."
        ) : (
          <>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Download Results
          </>
        )}
      </button>
    </GlassSurface>
  );
}

function ModelExecutionStatus({ 
  progress, running, done, error, t0Url, t1Url, midTruthUrl, generatedUrl, realGifUrl, interpolatedGifUrl
}: { 
  progress: number, running: boolean, done: boolean, error?: string | null,
  t0Url?: string | null, t1Url?: string | null, midTruthUrl?: string | null, generatedUrl?: string | null,
  realGifUrl?: string | null, interpolatedGifUrl?: string | null
}) {
  const steps = [
    { label: "Pre-processing", status: (done || progress >= 25) ? "checked" : (running ? "active" : "unchecked") },
    { label: "Optical Flow Estimation", status: (done || progress >= 60) ? "checked" : "unchecked" },
    { label: "Frame Synthesis", status: (done || progress >= 90) ? "checked" : "unchecked" },
    { label: "Post-processing", status: done ? "checked" : "unchecked" },
  ];

  return (
    <GlassSurface width="100%" height="auto" borderRadius={18} backgroundOpacity={0.08} blur={14} style={{ padding: "1.2rem 1.5rem", display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
        <div style={{ marginBottom: "-1rem" }}>
          <SectionTitle title="Model Execution Status" />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
          <div style={{ display: "flex", alignItems: "flex-end", gap: "0.5rem" }}>
            <span style={{ fontSize: 14, color: C.text }}>Overall Progress</span>
            <span style={{ fontSize: 14, color: C.text, fontFamily: "'JetBrains Mono', monospace" }}>{done ? 100 : running ? progress : 0}%</span>
          </div>
        </div>
      </div>
      
      {error && (
        <div style={{ color: C.red, fontSize: 12, padding: "0.5rem", background: "rgba(255,0,0,0.1)", borderRadius: 4 }}>
          Error: {error}
        </div>
      )}

      <div style={{ display: "flex", flexDirection: "row", gap: "1rem", alignItems: "center", flexWrap: "wrap", paddingLeft: "0.5rem" }}>
        {steps.map((step, idx) => {
          const isDone = step.status === "checked";
          return (
            <React.Fragment key={step.label}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.8rem" }}>
                {isDone ? (
                  <div style={{ width: 18, height: 18, background: "rgba(62, 224, 123, 0.15)", borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, border: `1px solid rgba(62, 224, 123, 0.3)` }}>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke={C.green} strokeWidth="3"><polyline points="20 6 9 17 4 12"/></svg>
                  </div>
                ) : step.status === "active" ? (
                  <div style={{ width: 18, height: 18, border: `2px solid ${C.orange}`, borderRadius: "50%", flexShrink: 0, boxShadow: `0 0 8px ${C.orange}` }} />
                ) : (
                  <div style={{ width: 18, height: 18, border: `2px solid ${C.dim}`, borderRadius: "50%", flexShrink: 0 }} />
                )}
                <span style={{ fontSize: 13, fontWeight: isDone ? 600 : 400, color: isDone ? C.green : C.text }}>{step.label}</span>
              </div>
              {idx < steps.length - 1 && (
                <div style={{ height: 2, flex: 1, minWidth: 15, background: isDone ? C.green : "rgba(255,255,255,0.1)", opacity: 0.5, borderRadius: 2 }} />
              )}
            </React.Fragment>
          );
        })}
      </div>
    </GlassSurface>
  );
}

export default function ResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { token } = useAuth();
  const files = location.state?.files as Record<string, File> | undefined;
  const fromZipUpload = location.state?.fromZipUpload as boolean | undefined;

  const [running, setRunning] = useState(true);
  const [progress, setProgress] = useState(0);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [t0Url, setT0Url] = useState<string | null>(null);
  const [t1Url, setT1Url] = useState<string | null>(null);
  const [midTruthUrl, setMidTruthUrl] = useState<string | null>(null);
  const [generatedUrl, setGeneratedUrl] = useState<string | null>(null);
  const [gapMapUrl, setGapMapUrl] = useState<string | null>(null);
  const [gapMinutes, setGapMinutes] = useState<number | null>(null);
  const [metrics, setMetrics] = useState<any>(null);
  
  const [realGifUrl, setRealGifUrl] = useState<string | null>(null);
  const [interpolatedGifUrl, setInterpolatedGifUrl] = useState<string | null>(null);
  const [diffMapUrl, setDiffMapUrl] = useState<string | null>(null);
  const [hsvFlowRealUrl, setHsvFlowRealUrl] = useState<string | null>(null);
  const [hsvFlowInterpolatedUrl, setHsvFlowInterpolatedUrl] = useState<string | null>(null);
  const [intensityGraphData, setIntensityGraphData] = useState<any>(null);
  const [cycloneEyeGraphData, setCycloneEyeGraphData] = useState<any>(null);
  const [trackingMetrics, setTrackingMetrics] = useState<any>(null);
  const [linearUrl, setLinearUrl] = useState<string | null>(null);
  const [pystepsUrl, setPystepsUrl] = useState<string | null>(null);
  const [rifeUrl, setRifeUrl] = useState<string | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);

  const [syncedRealGifUrl, setSyncedRealGifUrl] = useState<string | null>(null);
  const [syncedInterpolatedGifUrl, setSyncedInterpolatedGifUrl] = useState<string | null>(null);

  // Sync state for both looping animations
  const [animFrame, setAnimFrame] = useState(0);
  const [animPlaying, setAnimPlaying] = useState(true);
  const [animSpeed, setAnimSpeed] = useState(1);

  useEffect(() => {
    if (!animPlaying) return;
    const interval = setInterval(() => {
      setAnimFrame(prev => (prev + 1) % 3);
    }, 500 / animSpeed);
    return () => clearInterval(interval);
  }, [animPlaying, animSpeed]);

  useEffect(() => {
    if (realGifUrl && interpolatedGifUrl) {
      Promise.all([
        fetch(realGifUrl).then(res => res.blob()),
        fetch(interpolatedGifUrl).then(res => res.blob())
      ]).then(([realBlob, interpBlob]) => {
        // Set them at the exact same millisecond so they render and play synchronously
        const t = Date.now();
        setSyncedRealGifUrl(URL.createObjectURL(realBlob) + "#" + t);
        setSyncedInterpolatedGifUrl(URL.createObjectURL(interpBlob) + "#" + t);
      }).catch(err => {
        console.error("Failed to sync GIFs:", err);
        setSyncedRealGifUrl(realGifUrl);
        setSyncedInterpolatedGifUrl(interpolatedGifUrl);
      });
    } else {
      setSyncedRealGifUrl(realGifUrl);
      setSyncedInterpolatedGifUrl(interpolatedGifUrl);
    }
  }, [realGifUrl, interpolatedGifUrl]);

  const handleDownload = async () => {
    if (!done || isDownloading) return;
    setIsDownloading(true);
    const zip = new JSZip();
    const fetchToZip = async (url: string, filename: string) => {
      try {
        const response = await fetch(url);
        const blob = await response.blob();
        zip.file(filename, blob);
      } catch (e) {
        console.error("Failed to fetch", url, e);
      }
    };
    if (t0Url) await fetchToZip(t0Url, "t0_composite.png");
    if (t1Url) await fetchToZip(t1Url, "t1_composite.png");
    if (midTruthUrl) await fetchToZip(midTruthUrl, "ground_truth_midpoint.png");
    if (generatedUrl) await fetchToZip(generatedUrl, "generated_midpoint.png");
    if (realGifUrl) await fetchToZip(realGifUrl, "real_animation.gif");
    if (interpolatedGifUrl) await fetchToZip(interpolatedGifUrl, "interpolated_animation.gif");
    if (hsvFlowRealUrl) await fetchToZip(hsvFlowRealUrl, "hsv_flow_real.png");
    if (hsvFlowInterpolatedUrl) await fetchToZip(hsvFlowInterpolatedUrl, "hsv_flow_interpolated.png");
    const content = await zip.generateAsync({ type: "blob" });
    saveAs(content, "interpolation_results.zip");
    setIsDownloading(false);
  };

  useEffect(() => {
    if (!files && !fromZipUpload) {
      const saved = sessionStorage.getItem("experiment_results");
      if (saved) {
        const data = JSON.parse(saved);
        setT0Url(data.t0Url);
        setT1Url(data.t1Url);
        setMidTruthUrl(data.midTruthUrl);
        setGapMapUrl(data.gapMapUrl);
        setGapMinutes(data.gapMinutes);
        setGeneratedUrl(data.generatedUrl);
        setMetrics(data.metrics);
        setInterpolatedGifUrl(data.interpolatedGifUrl);
        setRealGifUrl(data.realGifUrl);
        setDiffMapUrl(data.diffMapUrl);
        setHsvFlowRealUrl(data.hsvFlowRealUrl);
        setHsvFlowInterpolatedUrl(data.hsvFlowInterpolatedUrl);
        setIntensityGraphData(data.intensityGraphData);
        setCycloneEyeGraphData(data.cycloneEyeGraphData);
        setTrackingMetrics(data.trackingMetrics);
        setLinearUrl(data.linearUrl || null);
        setPystepsUrl(data.pystepsUrl || null);
        setRifeUrl(data.rifeUrl || null);
        setProgress(100);
        setDone(true);
        setRunning(false);
      } else {
        setError("Missing required files (t0 and t1) and no previous results found.");
        setRunning(false);
      }
      return;
    }

    const runPipeline = async () => {
      try {
        let experiment_id: string;
        let uploadData: any;

        if (fromZipUpload) {
          experiment_id = transferData.experiment_id;
          uploadData = transferData.uploadData;
          setProgress(60);
        } else if (files) {
          setProgress(5);
          // 1. Create Experiment
          const createRes = await fetch("http://127.0.0.1:8000/experiments/?model_name=IFNET-GC&model_version=best_checkpoint", {
            method: "POST",
            headers: {
              "Authorization": `Bearer ${token}`
            }
          });
          if (!createRes.ok) throw new Error("Failed to create experiment");
          const createData = await createRes.json();
          experiment_id = createData.experiment_id;
          setProgress(25);

          // 2. Upload Files
          const formData = new FormData();
          formData.append("t0_tir", files.tirA);
          formData.append("t0_wv", files.wvA);
          formData.append("t1_tir", files.tirB);
          formData.append("t1_wv", files.wvB);
          if (files.tirC && files.wvC) {
            formData.append("tmid_tir", files.tirC);
            formData.append("tmid_wv", files.wvC);
          }

          const uploadRes = await fetch(`http://127.0.0.1:8000/experiments/${experiment_id}/upload`, {
            method: "POST",
            headers: {
              "Authorization": `Bearer ${token}`
            },
            body: formData
          });
          
          if (!uploadRes.ok) {
            const detail = await uploadRes.text();
            throw new Error(`Failed to upload files: ${detail}`);
          }
          uploadData = await uploadRes.json();
        } else {
          return;
        }

        // Apply pre-processing outputs
        setT0Url(uploadData.t0_url);
        setT1Url(uploadData.t1_url);
        setMidTruthUrl(uploadData.ground_truth_mid_url);
        setGapMapUrl(uploadData.gap_map_url);
        setGapMinutes(uploadData.gap_minutes);

        setProgress(60);

        // 3. Run Interpolation for main model (IFNET-GC)
        const runRes = await fetch(`http://127.0.0.1:8000/experiments/${experiment_id}/run?model_type=ifnet-gc`, {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${token}`
          }
        });
        if (!runRes.ok) {
          const detail = await runRes.text();
          throw new Error(`Failed to run interpolation: ${detail}`);
        }
        const runData = await runRes.json();

        setLinearUrl(null);
        setPystepsUrl(null);
        setRifeUrl(null);
        const fetchModel = async (modelType: string, setter: (val: string) => void) => {
           try {
             const res = await fetch(`http://127.0.0.1:8000/experiments/${experiment_id}/run?model_type=${modelType}`, {
               method: "POST",
               headers: { "Authorization": `Bearer ${token}` }
             });
             const data = await res.json();
             setter(data.generated_image_url);
           } catch (e) {
             console.error(`Error running ${modelType}:`, e);
           }
        };
        fetchModel("linear", (val) => {
          setLinearUrl(val);
          updateSessionStorage("linearUrl", val);
        });
        fetchModel("pysteps", (val) => {
          setPystepsUrl(val);
          updateSessionStorage("pystepsUrl", val);
        });
        fetchModel("rife", (val) => {
          setRifeUrl(val);
          updateSessionStorage("rifeUrl", val);
        });

        setGeneratedUrl(runData.generated_image_url);
        setMetrics(runData.ground_truth_metrics);
        setInterpolatedGifUrl(runData.interpolated_gif_url);
        setRealGifUrl(runData.real_gif_url);
        setDiffMapUrl(runData.difference_map_url);
        setHsvFlowRealUrl(runData.hsv_flow_real);
        setHsvFlowInterpolatedUrl(runData.hsv_flow_interpolated);
        setIntensityGraphData(runData.intensity_graph_data);
        setCycloneEyeGraphData(runData.cyclone_eye_graph_data);
        setTrackingMetrics(runData.tracking_metrics);
        
        setProgress(100);
        setDone(true);
        setRunning(false);
        
        const sessionData = {
          t0Url: uploadData.t0_url,
          t1Url: uploadData.t1_url,
          midTruthUrl: uploadData.ground_truth_mid_url,
          gapMapUrl: uploadData.gap_map_url,
          gapMinutes: uploadData.gap_minutes,
          generatedUrl: runData.generated_image_url,
          metrics: runData.ground_truth_metrics,
          interpolatedGifUrl: runData.interpolated_gif_url,
          realGifUrl: runData.real_gif_url,
          diffMapUrl: runData.difference_map_url,
          hsvFlowRealUrl: runData.hsv_flow_real,
          hsvFlowInterpolatedUrl: runData.hsv_flow_interpolated,
          intensityGraphData: runData.intensity_graph_data,
          cycloneEyeGraphData: runData.cyclone_eye_graph_data,
          trackingMetrics: runData.tracking_metrics,
        };
        sessionStorage.setItem("experiment_results", JSON.stringify(sessionData));
        
        function updateSessionStorage(key: string, val: string) {
          const current = JSON.parse(sessionStorage.getItem("experiment_results") || "{}");
          current[key] = val;
          sessionStorage.setItem("experiment_results", JSON.stringify(current));
        }
      } catch (err: any) {
        console.error(err);
        setError(err.message || "An unknown error occurred");
        setRunning(false);
      }
    };

    // To prevent double execution in strict mode when files change
    runPipeline();
  }, [files]);

  return (
    <div style={{
      fontFamily: "'Space Grotesk', sans-serif",
      background: "transparent",
      height: "100vh",
      overflow: "hidden",
      display: "flex",
      flexDirection: "column",
      boxSizing: "border-box",
    }}>
      <StatusBar running={running} done={done} progress={progress} />
      <div style={{ display: "flex", flex: 1, minHeight: 0, padding: "2rem", gap: "2rem" }}>
        {/* Left sidebar: nav + download button below */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem", alignItems: "center", flexShrink: 0 }}>
          <SidebarNav active="results" onNavigate={(r) => navigate(r)} />
          <button
            onClick={handleDownload}
            disabled={!done || isDownloading}
            title="Download Results"
            style={{
              width: 44, height: 44, borderRadius: "50%",
              background: done ? (isDownloading ? "rgba(255,255,255,0.05)" : "rgba(255,107,53,0.15)") : "rgba(255,255,255,0.03)",
              border: done ? "1px solid rgba(255,107,53,0.4)" : "1px solid rgba(255,255,255,0.06)",
              color: done ? "#FF6B35" : "#55555A",
              cursor: done ? "pointer" : "not-allowed",
              display: "flex", alignItems: "center", justifyContent: "center",
              transition: "all 0.2s",
              flexShrink: 0,
            }}
            onMouseEnter={e => { if (done && !isDownloading) { e.currentTarget.style.background = "rgba(255,107,53,0.25)"; e.currentTarget.style.borderColor = "rgba(255,107,53,0.6)"; } }}
            onMouseLeave={e => { if (done) { e.currentTarget.style.background = "rgba(255,107,53,0.15)"; e.currentTarget.style.borderColor = "rgba(255,107,53,0.4)"; } }}
          >
            {isDownloading ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ animation: "spin 1s linear infinite" }}>
                <path d="M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0"/>
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="7 10 12 15 17 10"/>
                <line x1="12" y1="15" x2="12" y2="3"/>
              </svg>
            )}
          </button>
        </div>
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "2rem", minWidth: 0, overflowY: "auto", paddingRight: "0.5rem" }}>
          <div style={{ flexShrink: 0 }}>
            <ModelExecutionStatus 
              progress={progress} running={running} done={done} error={error}
              t0Url={t0Url} t1Url={t1Url} midTruthUrl={midTruthUrl} generatedUrl={generatedUrl} 
              realGifUrl={realGifUrl} interpolatedGifUrl={interpolatedGifUrl}
            />
          </div>
          <div style={{ minHeight: 480, flexShrink: 0 }}>
            <TemporalSequence 
              t0Url={t0Url} t1Url={t1Url} midTruthUrl={midTruthUrl} generatedUrl={generatedUrl} gapMinutes={gapMinutes}
            />
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "2rem", flexShrink: 0 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem", height: "100%" }}>
              <div style={{ minHeight: 360, height: "100%" }}>
                <LoopingAnimation 
                  title="Looping Animation (Real)" 
                  frames={[t0Url, midTruthUrl, t1Url]} 
                  currentFrame={animFrame} setCurrentFrame={setAnimFrame}
                  isPlaying={animPlaying} setIsPlaying={setAnimPlaying}
                  speed={animSpeed} setSpeed={setAnimSpeed}
                />
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem", height: "100%" }}>
              <div style={{ minHeight: 360, height: "100%" }}>
                <LoopingAnimation 
                  title="Looping Animation (Interpolated)" 
                  frames={[t0Url, generatedUrl, t1Url]} 
                  currentFrame={animFrame} setCurrentFrame={setAnimFrame}
                  isPlaying={animPlaying} setIsPlaying={setAnimPlaying}
                  speed={animSpeed} setSpeed={setAnimSpeed}
                />
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem", height: "100%" }}>
              <div style={{ minHeight: 360, height: "100%" }}>
                <DifferenceMap src={diffMapUrl} metrics={metrics} />
              </div>
            </div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: (hsvFlowRealUrl || hsvFlowInterpolatedUrl) ? "1fr 2fr" : "1fr", gap: "2rem", flexShrink: 0, marginTop: "1rem" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: "2rem", minHeight: 340 }}>
              <ValidationMetrics metrics={metrics} />
            </div>
            {(hsvFlowRealUrl || hsvFlowInterpolatedUrl) && (
              <div style={{ minHeight: 340 }}>
                <MotionVectors realUrl={hsvFlowRealUrl} interpolatedUrl={hsvFlowInterpolatedUrl} />
              </div>
            )}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "2rem", flexShrink: 0, marginTop: "1rem", paddingBottom: "2rem" }}>
            <div style={{ minHeight: 460 }}>
              <IntensityChart data={intensityGraphData} />
            </div>
            <div style={{ minHeight: 460 }}>
              <EyeTrackingChart data={cycloneEyeGraphData} metrics={trackingMetrics} />
            </div>
            <div style={{ minHeight: 460 }}>
              {trackingMetrics && <CycloneTrackerMetrics tracking={trackingMetrics} />}
            </div>
          </div>
          
          {/* Model Comparison Section */}
          <div style={{ marginTop: "1rem", paddingBottom: "2rem" }}>
            <GlassSurface width="100%" height="auto" borderRadius={18} backgroundOpacity={0.08} blur={14} style={{ padding: "1.5rem", display: "flex", flexDirection: "column" }}>
              <SectionTitle title="Model Output Comparison" />
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1.5rem" }}>
                <ImagePlaceholder label="IFNET-GC (Current)" src={generatedUrl} />
                <ImagePlaceholder label="Linear Interpolation" src={linearUrl} />
                <ImagePlaceholder label="PySTEPS" src={pystepsUrl} />
                <ImagePlaceholder label="Pure RIFE" src={rifeUrl} />
              </div>
            </GlassSurface>
          </div>
        </div>
      </div>
    </div>
  );
}
