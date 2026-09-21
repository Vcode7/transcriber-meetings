import React from "react";
import { cn } from "@/lib/utils";

interface TapeMarqueeProps {
  text?: string;
  className?: string;
}

/** Handwritten tape banner with looping scroll. */
export function TapeMarquee({ text = "Work In Progress", className }: TapeMarqueeProps) {
  const items = Array.from({ length: 14 });
  return (
    <div
      className={cn(
        "relative overflow-hidden border-y-[2.5px] border-ink bg-accent/95 py-3 -rotate-1",
        className
      )}
    >
      <div className="flex animate-marquee whitespace-nowrap">
        {items.concat(items).map((_, i) => (
          <span
            key={i}
            className="mx-6 inline-flex items-center gap-4 font-display text-3xl font-bold text-accent-foreground"
          >
            <span>★</span>
            <span>{text}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

export default TapeMarquee;
