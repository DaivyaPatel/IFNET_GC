// import React, { useRef } from 'react';
// import { useFrame } from '@react-three/fiber';
// import { Stars, Sparkles } from '@react-three/drei';

// export default function Starfield() {
//   const starsRef = useRef();

//   useFrame((state, delta) => {
//     if (starsRef.current) {
//       // Extremely slow nebula/star movement
//       starsRef.current.rotation.y -= delta * 0.02;
//       starsRef.current.rotation.x -= delta * 0.01;
//     }
//   });

//   return (
//     <group ref={starsRef}>
//       <Stars 
//         radius={100} 
//         depth={50} 
//         count={3000} 
//         factor={3} 
//         saturation={0} 
//         fade 
//         speed={0.5} 
//       />
//       {/* Subtle blue nebula haze effect using sparkles */}
//       <Sparkles 
//         count={200} 
//         scale={50} 
//         size={10} 
//         speed={0.1} 
//         opacity={0.1} 
//         color="#45BFFF" 
//       />
//     </group>
//   );
// }

import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

const COUNT = 4000

const vertexShader = `
  attribute float aSize;
  attribute float aPhase;
  uniform float uTime;
  varying float vAlpha;
  void main() {
    vAlpha = 0.4 + 0.4 * sin(uTime * 0.3 + aPhase);
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = aSize * (250.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`

const fragmentShader = `
  varying float vAlpha;
  void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;
    float strength = 1.0 - (dist * 2.0);
    strength = pow(strength, 2.5);
    gl_FragColor = vec4(1.0, 1.0, 1.0, strength * vAlpha);
  }
`

// export default function Stars() {
//   const geometry = useMemo(() => {
//     const geo = new THREE.BufferGeometry()
//     const positions = new Float32Array(COUNT * 3)
//     const sizes = new Float32Array(COUNT)
//     const phases = new Float32Array(COUNT)

//     for (let i = 0; i < COUNT; i++) {
//       const radius = 25 + Math.random() * 60
//       const theta = Math.random() * Math.PI * 2
//       const phi = Math.acos(2 * Math.random() - 1)
//       positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta)
//       positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta)
//       positions[i * 3 + 2] = radius * Math.cos(phi)
//       sizes[i] = 0.5 + Math.random() * 2.5
//       phases[i] = Math.random() * Math.PI * 2
//     }

//     geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
//     geo.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1))
//     geo.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1))
//     return geo
//   }, [])

//   const uniforms = useMemo(() => ({ uTime: { value: 0 } }), [])

//   useFrame((state) => {
//     uniforms.uTime.value = state.clock.elapsedTime
//   })

//   return (
//     <points geometry={geometry}>
//       <shaderMaterial
//         vertexShader={vertexShader}
//         fragmentShader={fragmentShader}
//         uniforms={uniforms}
//         transparent
//         depthWrite={false}
//         blending={THREE.AdditiveBlending}
//       />
//     </points>
//   )
// }


//currently not in use since we are using a .jpg as starfield 