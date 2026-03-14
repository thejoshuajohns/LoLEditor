/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // LoL-inspired dark theme
        lol: {
          gold: '#C89B3C',
          'gold-light': '#F0E6D2',
          blue: '#0AC8B9',
          'blue-dark': '#0A1428',
          'blue-darker': '#091428',
          dark: '#010A13',
          'dark-light': '#1E2328',
          'dark-medium': '#1E282D',
          accent: '#C8AA6E',
          red: '#E84057',
          green: '#49B4A2',
          purple: '#9B59B6',
        },
      },
      fontFamily: {
        display: ['Avenir Next', 'Montserrat', 'Trebuchet MS', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-up': 'slideUp 0.3s ease-out',
      },
      keyframes: {
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
};
