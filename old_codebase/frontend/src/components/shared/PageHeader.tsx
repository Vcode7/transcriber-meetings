import { ReactNode } from "react";

import { cn } from "@/lib/utils";

interface PageHeaderProps {
  icon?: ReactNode;
  title: string;
  subtitle?: string;
  meta?: ReactNode;
  actions?: ReactNode;
  compact?: boolean;
}

export default function PageHeader({
  icon,
  title,
  subtitle,
  meta,
  actions,
  compact = false,
}: PageHeaderProps) {
  return (
    <div className={cn("panel-header", compact && "page-header-inline")}>
      {icon ? (
        <div
          aria-hidden="true"
          className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl border-2 border-accent/25 bg-accent/10 text-accent"
        >
          {icon}
        </div>
      ) : null}

      <div className="min-w-0 flex-1">
        <h1>{title}</h1>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>

      {meta ? <div className="flex-shrink-0">{meta}</div> : null}
      {actions ? <div className="flex flex-shrink-0 items-center gap-2">{actions}</div> : null}
    </div>
  );
}
