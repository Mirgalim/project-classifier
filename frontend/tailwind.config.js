/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["./src/**/*.{js,ts,jsx,tsx}"],
    theme: {
      extend: {
        colors: { bg: "#0b0b0b", card: "#121212", border: "#1f2937" },
        boxShadow: { soft: "0 6px 20px -6px rgba(0,0,0,.45)" },
        borderRadius: { xl: "0.75rem", '2xl': "1rem" }
      },
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui"] },
    },
    darkMode: "class",
    plugins: [require('@tailwindcss/forms')],
  }