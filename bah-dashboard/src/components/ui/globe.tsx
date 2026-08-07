import React, { useMemo } from "react";

interface GlobeProps {
  size?: number;
}

/* ─── STAR DENSITY CONFIG ─── */
const STAR_COUNTS = { small: 180, medium: 80, large: 35 };

type StarSize = "small" | "medium" | "large";
const STAR_SIZES: Record<StarSize, number> = { small: 2, medium: 4, large: 6 };

const TWINKLE_VARIANTS = [
  "twinkling",
  "twinkling-slow",
  "twinkling-long",
  "twinkling-fast",
];

function seededRandom(seed: string): number {
  let h = 0;
  for (let i = 0; i < seed.length; i++) {
    h = (h << 5) - h + seed.charCodeAt(i);
    h |= 0;
  }
  const x = Math.sin(h) * 10000;
  return x - Math.floor(x);
}

function generateStars(count: number, size: StarSize, prefix: string) {
  const stars = [];
  for (let i = 0; i < count; i++) {
    const seed = `${prefix}-${size}-${i}`;
    const left = (seededRandom(seed + "x") * 100).toFixed(2);
    const top = (seededRandom(seed + "y") * 100).toFixed(2);
    const animIdx = Math.floor(seededRandom(seed + "a") * TWINKLE_VARIANTS.length);
    const delay = (seededRandom(seed + "d") * 5).toFixed(2);
    const duration = (1.5 + seededRandom(seed + "t") * 4).toFixed(2);
    const opacity = (0.5 + seededRandom(seed + "o") * 0.5).toFixed(2);

    stars.push({
      key: seed,
      size: STAR_SIZES[size],
      left: `${left}%`,
      top: `${top}%`,
      animation: `${TWINKLE_VARIANTS[animIdx]} ${duration}s ease-in-out ${delay}s infinite`,
      opacity: parseFloat(opacity),
    });
  }
  return stars;
}

/* Cloud patch definitions: [left%, top%, width, height, opacity, blur] */
const CLOUD_PATCHES: [number, number, number, number, number, number][] = [
  [15, 20, 90, 50, 0.35, 14],
  [55, 12, 120, 70, 0.30, 18],
  [75, 35, 80, 45, 0.25, 12],
  [10, 55, 110, 60, 0.30, 16],
  [45, 48, 100, 55, 0.28, 14],
  [80, 65, 70, 40, 0.32, 10],
  [25, 78, 130, 65, 0.25, 20],
  [60, 82, 90, 50, 0.30, 12],
  [40, 30, 60, 35, 0.20, 10],
  [5, 40, 75, 45, 0.22, 15],
  [85, 50, 65, 35, 0.27, 11],
  [35, 62, 85, 48, 0.24, 13],
];

const Globe: React.FC<GlobeProps> = ({ size = 400 }) => {
  const orbitRadius = size * 0.58;
  const moonSize = Math.max(28, size * 0.07);

  const smallStars = useMemo(() => generateStars(STAR_COUNTS.small, "small", "bg"), []);
  const mediumStars = useMemo(() => generateStars(STAR_COUNTS.medium, "medium", "bg"), []);
  const largeStars = useMemo(() => generateStars(STAR_COUNTS.large, "large", "bg"), []);

  const renderStar = (s: ReturnType<typeof generateStars>[number]) => (
    <div
      key={s.key}
      className="absolute rounded-full bg-white"
      style={{
        width: s.size,
        height: s.size,
        left: s.left,
        top: s.top,
        animation: s.animation,
        opacity: s.opacity,
      }}
    />
  );

  return (
    <>
      <style>
        {`
          @keyframes earthRotate {
            0% { background-position: 0 0; }
            100% { background-position: 400px 0; }
          }
          @keyframes cloudRotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          @keyframes twinkling { 0%,100% { opacity:0.1; } 50% { opacity:1; } }
          @keyframes twinkling-slow { 0%,100% { opacity:0.1; } 50% { opacity:0.85; } }
          @keyframes twinkling-long { 0%,100% { opacity:0.05; } 50% { opacity:0.9; } }
          @keyframes twinkling-fast { 0%,100% { opacity:0.1; } 50% { opacity:1; } }
          @keyframes moonOrbit {
            0% { transform: translate(-50%, -50%) rotate(0deg) translateX(${orbitRadius}px) rotate(0deg); }
            100% { transform: translate(-50%, -50%) rotate(360deg) translateX(${orbitRadius}px) rotate(-360deg); }
          }
          @keyframes moonGlow {
            0%, 100% { box-shadow: 0 0 10px 2px rgba(200, 200, 220, 0.35); }
            50% { box-shadow: 0 0 20px 6px rgba(200, 200, 220, 0.55); }
          }
        `}
      </style>

      {/* Full viewport background layer */}
      <div className="fixed inset-0 overflow-hidden" style={{ zIndex: 0 }}>
        
        {/* Stars across the entire viewport */}
        {smallStars.map(renderStar)}
        {mediumStars.map(renderStar)}
        {largeStars.map(renderStar)}

        {/* Earth + Clouds container */}
        <div className="absolute" style={{ top: "50%", left: "50%", transform: "translate(-50%, -50%)" }}>
          {/* Earth surface */}
          <div
            className="relative rounded-full overflow-hidden"
            style={{
              width: size,
              height: size,
              backgroundImage: "url('https://pub-940ccf6255b54fa799a9b01050e6c227.r2.dev/globe.jpeg')",
              backgroundSize: "cover",
              backgroundPosition: "left",
              animation: "earthRotate 30s linear infinite",
              boxShadow: "0 0 20px rgba(255,255,255,0.2), -5px 0 8px #c3f4ff inset, 15px 2px 25px #000 inset, -24px -2px 34px #c3f4ff99 inset, 250px 0 44px #00000066 inset, 150px 0 38px #000000aa inset",
            }}
          >
            {/* Cloud layer — rotates slower than surface for parallax */}
            <div
              className="absolute inset-0 rounded-full"
              style={{
                animation: "cloudRotate 45s linear infinite",
                pointerEvents: "none",
              }}
            >
              {CLOUD_PATCHES.map(([l, t, w, h, o, b], i) => (
                <div
                  key={`cloud-${i}`}
                  className="absolute"
                  style={{
                    left: `${l}%`,
                    top: `${t}%`,
                    width: w,
                    height: h,
                    opacity: o,
                    filter: `blur(${b}px)`,
                    background: "radial-gradient(ellipse at center, rgba(255,255,255,0.95) 0%, rgba(240,245,255,0.6) 40%, transparent 75%)",
                    borderRadius: "50%",
                    transform: "translate(-50%, -50%)",
                  }}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Moon orbiting the Earth */}
        <div
          className="absolute"
          style={{
            top: "50%",
            left: "50%",
            width: moonSize,
            height: moonSize,
            borderRadius: "50%",
            background: "radial-gradient(circle at 35% 35%, #e8e8f0, #8a8a9e, #4a4a5e)",
            boxShadow: "0 0 12px 3px rgba(200, 200, 220, 0.4), inset -4px -4px 8px rgba(0,0,0,0.35)",
            animation: "moonOrbit 20s linear infinite, moonGlow 4s ease-in-out infinite",
            zIndex: 10,
          }}
        />
      </div>
    </>
  );
};

export default Globe;