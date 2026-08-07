import { useRef, useEffect } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Stars, Float } from "@react-three/drei";
import * as THREE from "three";

import { useGLTF } from "@react-three/drei";

// ── WebGL context cleanup helper ──────────────────────────────
function GLCleanup() {
  const { gl } = useThree();
  useEffect(() => {
    return () => {
      // Force-dispose the WebGL context when the Canvas unmounts
      // to prevent context exhaustion when navigating between pages
      gl.dispose();
    };
  }, [gl]);
  return null;
}

// ── Loaded Custom Satellite 3D Model Component ──────────────
function DetailedINSATModel({ scale = 1.6, autoRotate = true }: { scale?: number; autoRotate?: boolean }) {
  const groupRef = useRef<THREE.Group>(null);
  const { scene } = useGLTF("/satellite.glb");

  useFrame((_, delta) => {
    if (groupRef.current && autoRotate) {
      groupRef.current.rotation.y += delta * 0.25;
      groupRef.current.rotation.x = Math.sin(Date.now() * 0.0005) * 0.1;
    }
  });

  return (
    <group ref={groupRef} scale={scale} position={[0, -0.2, 0]}>
      <primitive object={scene} />
    </group>
  );
}

// ── Main exported component ───────────────────────────────────
interface SatelliteSceneProps {
  progress?: number;
  done?: boolean;
}

export default function SatelliteScene({ progress = 0 }: SatelliteSceneProps) {
  void progress;

  return (
    <div style={{ position: "relative", width: "100%", height: "100%", minHeight: "100%" }}>
      <Canvas
        camera={{ position: [0, 1.2, 5.5], fov: 45 }}
        style={{ background: "transparent", width: "100%", height: "100%" }}
        gl={{ antialias: true, alpha: true, powerPreference: "low-power" }}
        onCreated={({ gl }) => {
          gl.domElement.addEventListener('webglcontextlost', (e) => {
            e.preventDefault();
            console.warn("WebGL context lost — satellite scene will not render until context is restored.");
          });
        }}
      >
        <GLCleanup />
        {/* Lighting for dramatic space hardware look */}
        <ambientLight intensity={0.6} />
        <directionalLight position={[6, 8, 5]} intensity={2.2} color="#ffffff" />
        <directionalLight position={[-6, -4, -5]} intensity={0.8} color="#0055ff" />
        <pointLight position={[0, 4, 3]} intensity={1.5} color="#ffaa55" />

        {/* Stars background */}
        <Stars radius={60} depth={50} count={2500} factor={4} saturation={0} fade speed={0.8} />

        {/* Floating smooth animation for 3D INSAT Satellite */}
        <Float speed={1.5} rotationIntensity={0.2} floatIntensity={0.3}>
          <DetailedINSATModel scale={0.9} autoRotate={true} />
        </Float>

        <OrbitControls
          enableZoom={true}
          enablePan={false}
          autoRotate={false}
          minDistance={3.5}
          maxDistance={8.0}
        />
      </Canvas>

      {/* Badge overlay */}
      <div
        style={{
          position: "absolute",
          bottom: 10,
          left: 12,
          display: "flex",
          alignItems: "center",
          gap: 6,
          background: "rgba(6, 12, 26, 0.85)",
          backdropFilter: "blur(8px)",
          border: "1px solid rgba(255, 107, 53, 0.4)",
          borderRadius: 8,
          padding: "4px 10px",
          pointerEvents: "none",
        }}
      >
        <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#FF6B35" }} />
        <span style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, color: "#F0F4FF", letterSpacing: "0.08em" }}>
          INSAT-3DS SATELLITE 3D
        </span>
      </div>
    </div>
  );
}
