import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: process.env.NEURALIS_BASE_PATH || './',
  server: { port: 5173 },
  preview: { port: 4173 },
  build: {
    sourcemap: true,
    chunkSizeWarningLimit: 1400,
    rollupOptions: { output: { manualChunks: {
      react: ['react', 'react-dom', 'react-router-dom'],
      charts: ['recharts'],
      graphics: ['three', '@react-three/fiber', '@react-three/drei'],
    } } },
  },
});
