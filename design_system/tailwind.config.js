/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // Metabase Signature Royal Blue Palette
        brand: {
          50:  '#edf5ff',
          100: '#d6e7ff',
          200: '#b0d1ff',
          300: '#7ab3ff',
          400: '#509ee3', // Metabase signature royal light
          500: '#3b82f6', // Primary Action Blue
          600: '#2563eb', // Core Brand Blue
          700: '#1d4ed8', // Dark Hover
          800: '#1e40af', // Deep Slate Blue
          900: '#1e3a8a',
          950: '#111c44', // Dark Navy Accent
        },
        // Developer Slate Surfaces
        surface: {
          canvas: '#f8fafc',
          subtle: '#f1f5f9',
          card:   '#ffffff',
          dark:   '#0f172a',
          darker: '#0b0f19',
        },
        // Crisp Text Hierarchy
        content: {
          primary:   '#0f172a',
          secondary: '#475569',
          muted:     '#64748b',
          inverted:  '#ffffff',
        },
        // Border Tokens
        line: {
          subtle: '#f1f5f9',
          default: '#e2e8f0',
          strong:  '#cbd5e1',
          focus:   '#3b82f6',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['Fira Code', 'JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      boxShadow: {
        'crisp-sm': '0 1px 2px 0 rgba(15, 23, 42, 0.05)',
        'crisp-md': '0 4px 6px -1px rgba(15, 23, 42, 0.07), 0 2px 4px -2px rgba(15, 23, 42, 0.05)',
        'crisp-lg': '0 10px 15px -3px rgba(15, 23, 42, 0.08), 0 4px 6px -4px rgba(15, 23, 42, 0.04)',
        'brand-glow': '0 0 20px -3px rgba(59, 130, 246, 0.35)',
        'shiny-border': '0 0 0 1px rgba(255, 255, 255, 0.2) inset',
      },
      keyframes: {
        // Shimmer effect across gradient text/borders
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        // Rotating border gradient effect
        'border-rotate': {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
        // Subtle pulse for glowing indicator dots
        'pulse-glow': {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.6', transform: 'scale(1.1)' },
        },
      },
      animation: {
        shimmer: 'shimmer 3s infinite linear',
        'border-rotate': 'border-rotate 4s linear infinite',
        'pulse-glow': 'pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
};
