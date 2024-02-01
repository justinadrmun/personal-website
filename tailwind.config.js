module.exports = {
  content: ["./index.html", "./src/**/*.{jsx,js}"],
  darkmode: "class",
  theme: {
    extend: {
      fontFamily: {
          inter: ["inter", "serif"],
      },
      colors: {
        'stone-400': '#235977',
      },
  },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}

// /** @type {import('tailwindcss').Config} */
// module.exports = {
//   content: ["./src/**/*.{html,js,ts,jsx,tsx}"],
//   theme: {
//     extend: {},
//   },
//   plugins: [],
// };