import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere } from '@react-three/drei';
import * as THREE from 'three';

export default function Earth() {
  const earthRef = useRef();
  const atmosphereRef = useRef();

  useFrame((state, delta) => {
    if (earthRef.current) {
      earthRef.current.rotation.y += delta * 0.05;
    }
    if (atmosphereRef.current) {
      // Gentle breathing effect for atmosphere
      const scale = 1.05 + Math.sin(state.clock.elapsedTime * 2) * 0.005;
      atmosphereRef.current.scale.set(scale, scale, scale);
    }
  });

  return (
    <group position={[0, -8, -5]} scale={[1.5, 1.5, 1.5]}>
      {/* Main Earth Sphere */}
      <Sphere ref={earthRef} args={[6, 64, 64]}>
        <meshStandardMaterial 
          color="#0a192f" 
          roughness={0.6}
          metalness={0.1}
        />
      </Sphere>

      {/* Atmospheric Glow */}
      <Sphere ref={atmosphereRef} args={[6, 64, 64]}>
        <meshBasicMaterial 
          color="#45BFFF" 
          transparent={true} 
          opacity={0.15} 
          side={THREE.BackSide}
        />
      </Sphere>
    </group>
  );
}
