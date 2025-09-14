/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        bg: '#0b0e14',
        panel: '#11161f',
        muted: '#9aa4b2',
        text: '#e6edf3',
        primary: '#3b82f6',
        success: '#22c55e',
        danger: '#ef4444'
      }
    },
  },
  plugins: [],
}
