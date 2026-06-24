import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// During `pnpm dev` the C++ backend runs on :8000; proxy API calls to it.
export default defineConfig({
  plugins: [vue()],
  base: './',
  build: {
    outDir: 'dist',
    assetsInlineLimit: 0, // keep assets as separate files for clean embedding
  },
  server: {
    port: 5173,
    proxy: {
      '/file_parse': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/info': 'http://127.0.0.1:8000',
    },
  },
})
