/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // ITAC brand (override: ITAC blue + green, NOT ASU maroon/gold).
        itac: {
          blue: "#1F4695",
          "blue-dark": "#173472",
          "blue-light": "#E8EDF7",
          green: "#88C440",
          "green-dark": "#6FA032",
          "green-light": "#EEF6E2",
        },
      },
    },
  },
  plugins: [],
};
