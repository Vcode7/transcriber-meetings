import React from "react";
import { cn } from "@/lib/utils";

type Variant = "primary" | "secondary" | "ghost" | "danger";

interface SketchButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  asChild?: boolean;
}

export const SketchButton = React.forwardRef<HTMLButtonElement, SketchButtonProps>(
  ({ className, variant = "primary", children, ...props }, ref) => {
    const variantClass =
      variant === "primary"
        ? "btn-primary"
        : variant === "secondary"
        ? "btn-secondary"
        : variant === "ghost"
        ? "btn-ghost"
        : "btn-danger";
    return (
      <button ref={ref} className={cn("btn", variantClass, className)} {...props}>
        {children}
      </button>
    );
  }
);
SketchButton.displayName = "SketchButton";

export default SketchButton;
