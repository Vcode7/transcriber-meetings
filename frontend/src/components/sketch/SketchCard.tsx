import React from "react";
import { cn } from "@/lib/utils";

interface SketchCardProps extends React.HTMLAttributes<HTMLDivElement> {
  rotate?: number;
}

/**
 * Hand-drawn card with offset shadow. Use for content blocks.
 */
export function SketchCard({
  className,
  rotate = 0,
  style,
  children,
  ...props
}: SketchCardProps) {
  return (
    <div
      className={cn(
        "bg-card text-card-foreground border-[2.5px] border-ink",
        "shadow-sketch p-5 transition-transform",
        className
      )}
      style={{
        transform: rotate ? `rotate(${rotate}deg)` : undefined,
        borderRadius: "16px 22px 18px 24px / 22px 16px 24px 18px",
        ...style,
      }}
      {...props}
    >
      {children}
    </div>
  );
}

export default SketchCard;
