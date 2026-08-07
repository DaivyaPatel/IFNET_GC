import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Satellite } from 'lucide-react';
import './CTAButton.css';

export default function CTAButton() {
  const navigate = useNavigate();

  return (
    <button 
      className="cta-button"
      onClick={() => navigate('/dashboard')}
    >
      <div className="cta-content">
        <Satellite className="cta-icon" size={24} />
        <span className="cta-text font-display">VIEW DASHBOARD</span>
      </div>
      <div className="cta-arrow font-display">→</div>
      
      {/* HUD style borders */}
      <div className="cta-corner top-left"></div>
      <div className="cta-corner bottom-right"></div>
    </button>
  );
}
