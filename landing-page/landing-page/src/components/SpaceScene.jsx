// import React, { Suspense } from 'react';
// import { Canvas } from '@react-three/fiber';
// import { Environment, Preload } from '@react-three/drei';
// import { EffectComposer, Bloom } from '@react-three/postprocessing';
// import Starfield from './Starfield';

// export default function SpaceScene() {
//   return (
//     <Canvas
//       camera={{ position: [0, 0, 15], fov: 45 }}
//       dpr={[1, 2]}
//       gl={{ antialias: true, alpha: true }}
//     >
//       <color attach="background" args={['#02050A']} />
      
//       {/* Lighting */}
//       <ambientLight intensity={0.1} />
//       <directionalLight
//         position={[-10, 5, 5]}
//         intensity={1.5}
//         color="#ffffff"
//       />
//       <spotLight
//         position={[10, 10, -5]}
//         intensity={2}
//         color="#45BFFF"
//         angle={0.5}
//         penumbra={1}
//       />
      
//       <Suspense fallback={null}>
//         <Starfield />
//         <Environment preset="city" />
        
//         {/* Postprocessing for Stars Glow and Atmospheric effects */}
//         <EffectComposer>
//           <Bloom luminanceThreshold={0.2} luminanceSmoothing={0.9} height={300} intensity={1.5} />
//         </EffectComposer>
        
//         <Preload all />
//       </Suspense>
//     </Canvas>
//   );
// }


import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { Environment, Preload } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';

export default function SpaceScene() {
  return (
    <Canvas
      camera={{ position: [0, 0, 15], fov: 45 }}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true }}
    >
      <ambientLight intensity={0.1} />
      <directionalLight
        position={[-10, 5, 5]}
        intensity={1.5}
        color="#ffffff"
      />
      <spotLight
        position={[10, 10, -5]}
        intensity={2}
        color="#F2A23A"
        angle={0.5}
        penumbra={1}
      />
      <Suspense fallback={null}>
        <Environment preset="city" />
        <Preload all />
      </Suspense>
    </Canvas>
  );
}