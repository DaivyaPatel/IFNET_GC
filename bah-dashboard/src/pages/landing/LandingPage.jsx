import React, { useRef, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../../components/landing/Navbar';
import HeroContent from '../../components/landing/HeroContent';
import FeatureStrip from '../../components/landing/FeatureStrip';
import { SidebarNav, StatusBar, UniversalChannelViewer, UniversalCropMechanism, GlassSurface, C } from '../../components/ui/Dashboard';
import './LandingGlobal.css';
import './LandingPage.css';

export default function LandingPage() {
  const videoRef = useRef(null);
  const navigate = useNavigate();
  const [channelViewerResults, setChannelViewerResults] = useState(null);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.play().catch((err) => console.log('Video play error:', err));
    }
  }, []);
  
  return (
    <div className="landing-container">
      {/* 3D Background - Removed SpaceScene as it was empty and might crash on remount */}

      {/* Earth Video Layer */}
      <div className="earth-video-layer">
        <video 
          ref={videoRef}
          onTimeUpdate={() => {
            const video = videoRef.current;
            // If the video is within 0.15s of ending, instantly restart to hide the cut
            if (video && video.duration && video.currentTime >= video.duration - 0.15) {
              video.currentTime = 0;
              video.play().catch(() => {});
            }
          }}
          autoPlay 
          loop 
          muted 
          playsInline 
          className="earth-video"
        >
          <source src="/earth-globe-2.mp4" type="video/mp4" />
        </video>
      </div>

      {/* UI Overlay */}
      <div className="ui-layer" style={{ flexDirection: 'column' }}>
        <StatusBar running={false} done={false} progress={0} />
        <div style={{ display: 'flex', flexDirection: 'row', padding: '2rem', gap: '2rem', boxSizing: 'border-box', flex: 1 }}>
          <SidebarNav active="landing" onNavigate={(route, key) => navigate(route)} />
          <div className="main-content-wrapper" style={{ display: 'flex', flexDirection: 'column', flex: 1, position: 'relative', height: '100%' }}>
            <Navbar />
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
              <HeroContent />
              
              <div style={{ display: 'flex', flexDirection: 'row', gap: '2rem', padding: '2rem', flexWrap: 'wrap' }}>
                {/* Left Column: Features (2x2) */}
                <div style={{ flex: 1, minWidth: '400px', display: 'flex', alignItems: 'center' }}>
                  <FeatureStrip />
                </div>
                
                {/* Right Column: Universal Tools Section */}
                <div style={{ flex: 1.5, display: 'flex', gap: '1rem', minWidth: '600px' }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <UniversalChannelViewer onResults={setChannelViewerResults} />
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <UniversalCropMechanism />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Modal for Results - rendered at root level */}
      {channelViewerResults && (
        <div style={{ position: "fixed", inset: 0, zIndex: 9999, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.8)", backdropFilter: "blur(10px)", padding: "2rem" }}>
          <GlassSurface width="auto" height="auto" borderRadius={20} backgroundOpacity={0.15} blur={20} style={{ display: "flex", flexDirection: "column", border: `1px solid rgba(255,255,255,0.1)`, padding: "1.5rem", overflow: "hidden", maxWidth: "500px", minWidth: "320px", maxHeight: "80vh" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem", flexShrink: 0, gap: "2rem" }}>
              <div>
                <h2 style={{ margin: 0, color: C.text, fontSize: 18 }}>Extracted Data</h2>
                <p style={{ margin: "0.2rem 0 0 0", color: C.dim, fontSize: 12 }}>Generated output preview.</p>
              </div>
              <button onClick={() => setChannelViewerResults(null)} className="cursor-target" style={{ background: "rgba(255,255,255,0.1)", border: "none", color: C.text, padding: "0.5rem 1rem", borderRadius: C.radiusSm, cursor: "pointer", fontWeight: "bold", fontSize: 12 }}>Close</button>
            </div>
            
            <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: "1rem" }}>
              {channelViewerResults.map((item, i) => (
                <div key={i} style={{ display: "flex", flexDirection: "column", gap: "0.8rem", background: "rgba(0,0,0,0.4)", padding: "0.8rem", borderRadius: C.radiusSm, border: `1px solid rgba(255,255,255,0.05)` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: C.cyan }}>{item.channel_name}</div>
                    {item.url && !item.filename?.endsWith(".nc") && (
                      <a href={item.url} download={item.filename || "extracted_crop.png"} className="cursor-target" style={{ background: C.orange, color: "#000", padding: "0.3rem 0.6rem", borderRadius: 4, textDecoration: "none", fontWeight: "bold", fontSize: 10 }}>
                        Download Image
                      </a>
                    )}
                  </div>
                  
                  {item.filename?.endsWith(".nc") ? (
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.8rem", alignItems: "center", justifyContent: "center", height: "100px", background: "rgba(0,0,0,0.6)", borderRadius: 8 }}>
                      <div style={{ color: C.dim, fontSize: 11 }}>Raw .nc data</div>
                      <a href={item.url} target="_blank" download className="cursor-target" style={{ background: C.green, color: "#000", padding: "0.4rem 0.8rem", borderRadius: 4, textDecoration: "none", fontWeight: "bold", fontSize: 11 }}>
                        Download .nc File
                      </a>
                    </div>
                  ) : (
                    <img src={item.url} alt={item.channel_name} style={{ width: "100%", height: "auto", borderRadius: 8, objectFit: "contain", maxHeight: 300, background: "#000" }} />
                  )}
                </div>
              ))}
            </div>
          </GlassSurface>
        </div>
      )}
    </div>
  );
}
