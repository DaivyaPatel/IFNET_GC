import React from 'react';
import './Navbar.css';

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="nav-left">
        {/* Minimal futuristic orbital-ring logo */}
        <div className="orbital-logo">
          <div className="ring ring-outer"></div>
          <div className="ring ring-inner"></div>
          <div className="core"></div>
        </div>
        <span className="team-name font-display">THE FELLOWSHIP OF THE RING</span>
      </div>
      
      <div className="nav-right">
        <a href="#home" className="nav-link font-display">HOME</a>
        <a href="#about" className="nav-link font-display">ABOUT</a>
        <a href="#team" className="nav-link font-display">TEAM</a>
        <a href="#contact" className="nav-link font-display">CONTACT</a>
      </div>
    </nav>
  );
}
