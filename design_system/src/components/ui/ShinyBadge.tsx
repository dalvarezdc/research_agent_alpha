import React from 'react';

export interface ShinyBadgeProps {
  variant?: 'iridescent' | 'emerald' | 'amber' | 'blue-glow';
  pulse?: boolean;
  icon?: React.ReactNode;
  fontMono?: boolean;
  children: React.ReactNode;
  className?: string;
}

export const ShinyBadge: React.FC<ShinyBadgeProps> = ({
  variant = 'iridescent',
  pulse = false,
  icon,
  fontMono = false,
  children,
  className = '',
}) => {
  const fontClass = fontMono ? 'font-mono' : 'font-sans';

  if (variant === 'iridescent') {
    return (
      <div className={`shiny-border-wrapper inline-block ${className}`}>
        <div className={`shiny-border-inner ${fontClass}`}>
          {pulse && (
            <span className="w-2 h-2 rounded-full bg-purple-500 animate-pulse-glow" />
          )}
          {icon && <span className="text-purple-600 dark:text-purple-400">{icon}</span>}
          <span className="text-shimmer">{children}</span>
        </div>
      </div>
    );
  }

  if (variant === 'emerald') {
    return (
      <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-300/60 shadow-[0_0_12px_rgba(16,185,129,0.2)] dark:bg-emerald-950/40 dark:text-emerald-300 dark:border-emerald-800 ${fontClass} ${className}`}>
        {pulse && <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />}
        {icon}
        <span>{children}</span>
      </span>
    );
  }

  if (variant === 'amber') {
    return (
      <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-50 text-amber-800 border border-amber-300/70 shadow-[0_0_10px_rgba(245,158,11,0.15)] dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-800 ${fontClass} ${className}`}>
        {pulse && <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />}
        {icon}
        <span>{children}</span>
      </span>
    );
  }

  // Default: Blue Glow (Metabase Signature Accent)
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-brand-50 text-brand-700 border border-brand-200 shadow-[0_0_14px_rgba(59,130,246,0.25)] dark:bg-brand-950/50 dark:text-brand-300 dark:border-brand-800 ${fontClass} ${className}`}>
      {pulse && <span className="w-1.5 h-1.5 rounded-full bg-brand-500 animate-pulse" />}
      {icon}
      <span>{children}</span>
    </span>
  );
};
