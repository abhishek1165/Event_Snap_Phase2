/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      /* map radius scale to the --radius token */
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 4px)',
        sm: 'calc(var(--radius) - 8px)',
        xl: 'calc(var(--radius) + 4px)',
        '2xl': 'calc(var(--radius) + 10px)',
      },
      colors: {
        /* ── semantic surfaces / text (our Event Snap tokens) ── */
        ink:        'var(--ink)',
        'ink-2':    'var(--ink-2)',
        surface: {
          DEFAULT:  'var(--surface)',
          2:        'var(--surface-2)',
          3:        'var(--surface-3)',
        },
        text: {
          DEFAULT:  'var(--text)',
          dim:      'var(--text-dim)',
          faint:    'var(--text-faint)',
        },
        frame: {
          DEFAULT:  'var(--border)',
          soft:     'var(--border-soft)',
        },
        iris: {            // brand violet (avoid clash with shadcn `accent`)
          DEFAULT:  'var(--accent)',
          2:        'var(--accent-2)',
        },
        match:      'var(--match)',
        memory:     'var(--memory)',

        /* ── shadcn compatibility (keep ui/* components working) ── */
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))'
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))'
        },
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))'
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))'
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))'
        },
        accent: {
          DEFAULT: 'hsl(var(--accent-hsl))',
          foreground: 'hsl(var(--accent-fg))'
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))'
        },
        border: 'hsl(var(--border-hsl))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring-hsl))',
        chart: {
          '1': 'hsl(var(--chart-1))',
          '2': 'hsl(var(--chart-2))',
          '3': 'hsl(var(--chart-3))',
          '4': 'hsl(var(--chart-4))',
          '5': 'hsl(var(--chart-5))'
        }
      },
      fontFamily: {
        display: ['Instrument Sans', 'system-ui', 'sans-serif'],
        body:    ['Inter', 'system-ui', 'sans-serif'],
        mono:    ['IBM Plex Mono', 'monospace'],
      },
      keyframes: {
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' }
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' }
        },
        'fade-in': {
          from: { opacity: '0' },
          to: { opacity: '1' }
        },
        'shimmer': {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' }
        },
        'aurora': {
          '0%,100%': { transform: 'translate3d(0,0,0) scale(1)' },
          '50%':     { transform: 'translate3d(2%, -3%, 0) scale(1.05)' }
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
        'fade-in': 'fade-in 0.4s ease-out',
        'shimmer': 'shimmer 2.2s linear infinite',
        'aurora': 'aurora 16s ease-in-out infinite',
      },
      backgroundImage: {
        'brand-gradient': 'linear-gradient(135deg,#6d5ef5 0%,#8b7bff 100%)',
        'mesh-hero': 'radial-gradient(60% 80% at 50% 0%,rgba(109,94,245,.22),transparent 70%),radial-gradient(50% 60% at 85% 100%,rgba(52,211,153,.12),transparent 70%)',
        'edge-light': 'linear-gradient(180deg,rgba(255,255,255,.12) 0%,rgba(255,255,255,.02) 100%)',
      },
    }
  },
  plugins: [require("tailwindcss-animate")],
};
