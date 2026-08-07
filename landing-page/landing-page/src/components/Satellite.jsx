import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Box, Cylinder, Html } from '@react-three/drei';
import * as THREE from 'three';

export default function Satellite() {
  const satRef = useRef();

  useFrame((state, delta) => {
    if (satRef.current) {
      // Subtle orbital floating motion
      satRef.current.position.y += Math.sin(state.clock.elapsedTime * 0.5) * 0.002;
      satRef.current.rotation.y += delta * 0.05;
      satRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.2) * 0.02;
    }
  });

  return (
    <group position={[3.5, 2.5, 0]} ref={satRef} rotation={[0.2, -0.4, 0]}>
      {/* Main Body (Metallic/Gold) */}
      <Box args={[1, 1.2, 1]}>
        <meshStandardMaterial color="#888888" metalness={0.8} roughness={0.2} />
      </Box>

      {/* Gold thermal insulation details */}
      <Box args={[1.05, 0.8, 1.05]} position={[0, -0.1, 0]}>
        <meshStandardMaterial color="#d4af37" metalness={0.6} roughness={0.4} />
      </Box>

      {/* Solar Panel Left */}
      <Box args={[2.5, 0.05, 0.8]} position={[-1.8, 0, 0]}>
        <meshStandardMaterial color="#0033aa" metalness={0.9} roughness={0.1} />
      </Box>

      {/* Solar Panel Right */}
      <Box args={[2.5, 0.05, 0.8]} position={[1.8, 0, 0]}>
        <meshStandardMaterial color="#0033aa" metalness={0.9} roughness={0.1} />
      </Box>

      {/* Reflector Dish */}
      <Cylinder args={[0.5, 0.1, 0.2, 32]} position={[0, 0.7, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <meshStandardMaterial color="#aaaaaa" metalness={0.5} roughness={0.5} />
      </Cylinder>
      
      {/* Sensor/Antenna */}
      <Cylinder args={[0.02, 0.02, 0.8]} position={[0.4, 0.8, 0.2]}>
        <meshStandardMaterial color="#ffffff" metalness={0.8} roughness={0.2} />
      </Cylinder>

      {/* HUD Annotation */}
      <Html position={[-2, 1, 0]} center>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          color: '#F2A23A',
          fontFamily: "'Inter', sans-serif",
          pointerEvents: 'none'
        }}>
        <div style={{
          width: '40px',
          height: '1px',
          backgroundColor: '#F2A23A',
          boxShadow: '0 0 5px #F2A23A'
        }}></div>
        <div>
          <div style={{ fontSize: '12px', fontWeight: 'bold', letterSpacing: '0.1em' }}>INSAT-3D</div>
          <div style={{ fontSize: '8px', color: '#A8B4C2', letterSpacing: '0.05em' }}>INDIA'S METEOROLOGICAL SATELLITE</div>
        </div>
      </div>
      </Html>
    </group>
  );
}
