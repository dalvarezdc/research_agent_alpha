import React from 'react';

export interface CardProps {
  title?: string;
  subtitle?: string;
  badge?: React.ReactNode;
  interactive?: boolean;
  children: React.ReactNode;
  footer?: React.ReactNode;
  className?: string;
}

export const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  badge,
  interactive = false,
  children,
  footer,
  className = '',
}) => {
  return (
    <div
      className={`
        bg-white dark:bg-slate-900 
        border border-line-default dark:border-slate-800 
        rounded-xl shadow-crisp-sm transition-all duration-200 
        ${interactive ? 'hover:border-brand-300 hover:shadow-crisp-md hover:-translate-y-0.5 cursor-pointer dark:hover:border-brand-600/50' : ''}
        ${className}
      `}
    >
      {(title || badge) && (
        <div className="flex items-center justify-between px-6 pt-5 pb-3 border-b border-slate-100 dark:border-slate-800/60">
          <div>
            {title && <h3 className="text-base font-semibold text-content-primary dark:text-white">{title}</h3>}
            {subtitle && <p className="text-xs text-content-muted mt-0.5">{subtitle}</p>}
          </div>
          {badge && <div>{badge}</div>}
        </div>
      )}
      <div className="p-6">{children}</div>
      {footer && (
        <div className="px-6 py-3.5 bg-slate-50/70 dark:bg-slate-800/40 border-t border-line-default dark:border-slate-800 rounded-b-xl">
          {footer}
        </div>
      )}
    </div>
  );
};
