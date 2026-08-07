// import React from 'react';
// import CTAButton from './CTAButton';
// import './HeroContent.css';

// function AnimatedText({ text, startDelay = 0 }) {
//   return text.split('').map((char, i) => (
//     <span
//       key={i}
//       className="letter"
//       style={{ animationDelay: `${startDelay + i * 0.06}s` }}
//     >
//       {char}
//     </span>
//   ));
// }

// export default function HeroContent() {
//   return (
//     <div className="hero-content">

//       <div className="hero-left-zone">
//         <h1 className="hero-title font-display">
//           <span className="title-white">
//             <AnimatedText text="FRAME" startDelay={0} />
//           </span>
//           <br />
//           <span className="title-gradient">
//             <AnimatedText text="INTERPOLATION" startDelay={0.25} />
//           </span>
//         </h1>

//         <p className="hero-description">
//           Advanced AI-powered frame interpolation<br />
//           for satellite imagery.
//         </p>

//         <div className="cta-wrapper">
//           <CTAButton />
//         </div>
//       </div>

//       <div className="hero-right-zone">
//         <img src="/satellite.png" alt="Satellite" className="satellite-image" />
//       </div>

//     </div>
//   );
// }


import React, { useState, useRef, useEffect } from 'react';
import CTAButton from './CTAButton';
import './HeroContent.css';

function AnimatedText({ text, startDelay = 0 }) {
  return text.split('').map((char, i) => (
    <span
      key={i}
      className="letter"
      style={{ animationDelay: `${startDelay + i * 0.08}s` }}
    >
      {char}
    </span>
  ));
}

export default function HeroContent() {
  const [showSatelliteInfo, setShowSatelliteInfo] = useState(false);
  const satelliteRef = useRef(null);



  return (
    <div className="hero-content">

      <div className="hero-left-zone">
        <h1 className="hero-title font-display" style={{ fontFamily: "'Aquire', sans-serif", fontWeight: 900 }}>
          <span className="title-white">
            <AnimatedText text="IFNET-GC" startDelay={0} />
          </span>
        </h1>

        <p className="hero-description">
          Time-aware. Storm-trained. Gap-conditioned<br />
          Optical flow that scales with time
        </p>

        <div className="cta-wrapper">
          <CTAButton />
        </div>
      </div>

      <div
        className="hero-right-zone"
        ref={satelliteRef}
        onMouseEnter={() => setShowSatelliteInfo(true)}
        onMouseLeave={() => setShowSatelliteInfo(false)}
        style={{ pointerEvents: 'auto' }}
      >
        <img
          src="/satellite.png"
          alt="Satellite"
          className="satellite-image cursor-target"
        />

        {showSatelliteInfo && (
          <div className="satellite-info-panel">
            <div className="satellite-info-header">INSAT-3D</div>
            <div className="satellite-info-row">
              <span className="info-label">Developed by</span>
              <span className="info-value">Indian Space Research Organisation (ISRO)</span>
            </div>
            <div className="satellite-info-row">
              <span className="info-label">Type</span>
              <span className="info-value">Meteorological / Earth-observation satellite</span>
            </div>
            <div className="satellite-info-row">
              <span className="info-label">Primary purpose</span>
              <span className="info-value">Weather monitoring, forecasting, and disaster warning</span>
            </div>
            <div className="satellite-info-row">
              <span className="info-label">Launch date</span>
              <span className="info-value">26 July 2013</span>
            </div>
            <div className="satellite-info-row">
              <span className="info-label">Launch vehicle</span>
              <span className="info-value">Ariane-5 VA-214</span>
            </div>
            <div className="satellite-info-row">
              <span className="info-label">Orbit</span>
              <span className="info-value">Geostationary orbit at approximately 82° E longitude</span>
            </div>
            <div className="satellite-info-row">
              <span className="info-label">Launch mass</span>
              <span className="info-value">2,060 kg</span>
            </div>
            <div className="satellite-info-row">
              <span className="info-label">Dimensions</span>
              <span className="info-value">2.4 × 1.6 × 1.5 m</span>
            </div>
          </div>
        )}
      </div>

    </div>
  );
}