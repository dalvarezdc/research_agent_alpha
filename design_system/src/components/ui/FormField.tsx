import React from 'react';

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  error?: boolean;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className = '', error, ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={`
          w-full px-3.5 py-2 text-sm rounded-lg bg-white dark:bg-slate-900 
          text-content-primary dark:text-white placeholder:text-slate-400 
          border ${error ? 'border-red-500 focus:ring-red-500/30' : 'border-line-default hover:border-slate-300 focus:border-brand-500 focus:ring-brand-500/30 dark:border-slate-800 dark:hover:border-slate-700'} 
          shadow-crisp-sm focus:outline-none focus:ring-2 transition-all duration-150
          ${className}
        `}
        {...props}
      />
    );
  }
);
Input.displayName = 'Input';

export interface FormFieldProps {
  label: string;
  hint?: string;
  error?: string;
  required?: boolean;
  children: React.ReactNode;
  className?: string;
}

export const FormField: React.FC<FormFieldProps> = ({
  label,
  hint,
  error,
  required,
  children,
  className = '',
}) => {
  return (
    <div className={`space-y-1.5 ${className}`}>
      <div className="flex justify-between items-center">
        <label className="text-xs font-semibold uppercase tracking-wider text-content-secondary dark:text-slate-300">
          {label} {required && <span className="text-brand-500">*</span>}
        </label>
        {hint && <span className="text-xs text-content-muted">{hint}</span>}
      </div>
      {children}
      {error && <p className="text-xs text-red-600 dark:text-red-400 mt-1">{error}</p>}
    </div>
  );
};
