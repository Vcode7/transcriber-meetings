import React from "react";

/**
 * Global SVG defs for the sketchy/hand-drawn aesthetic.
 * Mount once near the app root. Used by `.squiggle`, `.sketch-border`, etc.
 */
export function SquiggleFilter() {
  return (
    <svg
      aria-hidden="true"
      width="0"
      height="0"
      style={{ position: "absolute", width: 0, height: 0 }}
    >
      <defs>
        <filter id="squiggle">
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.02"
            numOctaves="3"
            result="noise"
            seed="2"
          />
          <feDisplacementMap in="SourceGraphic" in2="noise" scale="2.2" />
        </filter>
        <filter id="squiggle-soft">
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.015"
            numOctaves="2"
            result="noise"
            seed="5"
          />
          <feDisplacementMap in="SourceGraphic" in2="noise" scale="1.2" />
        </filter>
        <filter id="paper-rough">
          <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" />
          <feColorMatrix values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 .08 0" />
          <feComposite operator="in" in2="SourceGraphic" />
          <feComposite operator="over" in2="SourceGraphic" />
        </filter>
      </defs>
    </svg>
  );
}

export default SquiggleFilter;
