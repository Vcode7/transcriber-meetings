import React from "react";
import { cn } from "@/lib/utils";

type StickyColor = "yellow" | "blue" | "pink" | "green" | "indigo" | "red";

const COLOR_MAP: Record<StickyColor, string> = {
  yellow: "bg-sticky-yellow",
  blue: "bg-sticky-blue",
  pink: "bg-sticky-pink",
  green: "bg-sticky-green",
  indigo: "bg-sticky-indigo",
  red: "bg-sticky-red",
};

interface StickyNoteProps extends React.HTMLAttributes<HTMLDivElement> {
  color?: StickyColor;
  rotate?: number;
  tape?: boolean;
}

export function StickyNote({
  color = "yellow",
  rotate = 0,
  tape = true,
  className,
  style,
  children,
  ...props
}: StickyNoteProps) {
  return (
    <div
      className={cn(
        "relative p-5 text-ink shadow-sketch border-2 border-ink/80",
        "font-hand transition-transform duration-200 hover:-rotate-1",
        COLOR_MAP[color],
        tape && "tape",
        className
      )}
      style={{
        transform: `rotate(${rotate}deg)`,
        borderRadius: "10px 18px 12px 16px / 16px 12px 18px 10px",
        ...style,
      }}
      {...props}
    >
      {children}
    </div>
  );
}

export default StickyNote;
