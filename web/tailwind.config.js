import colors from 'tailwindcss/colors.js';

/**
 * Theme-aware colours. The UI was written dark-first (text-white, bg-neutral-900, text-purple-300...),
 * so instead of doubling every class with a light variant, the colours that carry the theme are CSS
 * variables (see index.css): neutrals and white flip, and the light accent shades used for text
 * on dark surfaces turn into their darker counterparts in the light theme.
 */
const v = name => `rgb(var(--c-${name}) / <alpha-value>)`;
const scale = (name, shades) => Object.fromEntries(shades.map(s => [s, v(`${name}-${s}`)]));
const NEUTRAL = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950];
const ACCENT = [200, 300, 400, 950];
const accent = name => ({ ...colors[name], ...scale(name, ACCENT) });

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: ['selector', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        white: v('white'),
        neutral: scale('neutral', NEUTRAL),
        purple: accent('purple'),
        rose: accent('rose'),
        amber: accent('amber'),
        emerald: accent('emerald'),
        sky: accent('sky'),
        orange: accent('orange'),
        app: v('app'),
        panel: v('panel'),
      },
    },
  },
  plugins: [],
};
