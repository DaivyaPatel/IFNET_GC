import React from 'react';
import { Layers, Sparkles, Target, BarChart2 } from 'lucide-react';
import './FeatureStrip.css';

export default function FeatureStrip() {
  const features = [
    {
      icon: <Layers size={32} color="#F2A23A" />,
      title: 'FILL GAPS',
      description: 'Bridge temporal gaps\nin satellite sequences'
    },
    {
      icon: <Sparkles size={32} color="#4FBF73" />,
      title: 'AI-POWERED',
      description: 'Deep learning models for\nhigh-fidelity interpolation'
    },
    {
      icon: <Target size={32} color="#5D8FAF" />,
      title: 'PRECISE',
      description: 'High PSNR & SSIM for\naccurate reconstruction'
    },
    {
      icon: <BarChart2 size={32} color="#F2A23A" />,
      title: 'VISUALIZE',
      description: 'Compare and analyze\nresults in real-time'
    }
  ];

  return (
    <div className="feature-strip">
      {features.map((feature, index) => (
        <React.Fragment key={index}>
          <div className="feature-item">
            <div className="feature-icon">{feature.icon}</div>
            <div className="feature-content">
              <h3 className="feature-title font-display">{feature.title}</h3>
              <p className="feature-description">{feature.description}</p>
            </div>
          </div>
          {index < features.length - 1 && <div className="feature-divider"></div>}
        </React.Fragment>
      ))}
    </div>
  );
}