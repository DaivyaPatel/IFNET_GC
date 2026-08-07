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
      
    </nav>
  );
}
