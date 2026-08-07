import React from 'react';
import './GlassSurface.css';

const GlassSurface = ({
  children,
  width = '100%',
  height = '100%',
  borderRadius = 20,
  className = '',
  style = {},
  // Ignoring the old complex SVG filter props for the new Mist effect
  ...rest
}) => {
  const containerStyle = {
    ...style,
    width: typeof width === 'number' ? `${width}px` : width,
    height: typeof height === 'number' ? `${height}px` : height,
    borderRadius: `${borderRadius}px`,
  };

  return (
    <div
      className={`glass-surface ${className}`}
      style={containerStyle}
      {...rest}
    >
      <div className="glass-surface__content">{children}</div>
    </div>
  );
};

export default GlassSurface;
