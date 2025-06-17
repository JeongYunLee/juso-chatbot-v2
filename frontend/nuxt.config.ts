// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: "2024-04-03",
  devtools: { enabled: true },
  modules: [
    "@nuxtjs/tailwindcss",
  ],
  css: ["prismjs/themes/prism.css"],
  app: {
    baseURL: '/projects/chatbot/',
  },
  runtimeConfig: {
    public: {
      apiUrl: process.env.VITE_API_URL 
    }
  }
});
