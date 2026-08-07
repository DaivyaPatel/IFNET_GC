import React from 'react';
import SpaceScene from '../components/SpaceScene';
import Navbar from '../components/Navbar';
import HeroContent from '../components/HeroContent';
import FeatureStrip from '../components/FeatureStrip';
import './LandingPage.css'; // Optional local styles if needed

export default function LandingPage() {
  return (
    <div className="landing-container">
      {/* 3D Background */}
      <div className="scene-container">
        <SpaceScene />
      </div>

      {/* Earth Video Layer */}
      <div className="earth-video-layer">
        <video autoPlay loop muted playsInline className="earth-video">
          <source src="/earth-globe-2.mp4" type="video/mp4" />
        </video>
      </div>

      {/* UI Overlay */}
      <div className="ui-layer">
        <Navbar />
        <HeroContent />
        <FeatureStrip />
      </div>
    </div>
  );
}
