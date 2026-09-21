import React from "react";

/**
 * SmoothScroll wrapper - DISABLED to fix mouse wheel scrolling issues.
 * 
 * The Lenis smooth scroll library was intercepting native scroll events
 * and preventing mouse wheel, touchpad, and keyboard scrolling from working.
 * Native browser scrolling is now used instead.
 */
export function SmoothScroll({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

export default SmoothScroll;
