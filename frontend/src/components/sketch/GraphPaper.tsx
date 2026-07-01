import React from "react";
import { cn } from "@/lib/utils";

/** Faux graph-paper background panel. */
export function GraphPaper({
  className,
  children,
}: {
  className?: string;
  children?: React.ReactNode;
}) {
  return <div className={cn("bg-graph", className)}>{children}</div>;
}

export default GraphPaper;
