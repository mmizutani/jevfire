import { defineConfig } from 'vite';
const jevTarget = process.env.JEVFIRE_JEV_URL || 'http://127.0.0.1:8011';
const jevApiKey = process.env.JEVFIRE_JEV_API_KEY;
if (
  jevApiKey &&
  !['127.0.0.1', 'localhost', '[::1]'].includes(new URL(jevTarget).hostname)
)
  throw new Error('Authenticated Jev proxy requires a loopback target');
const jevHeaders = jevApiKey ? { Authorization: `Bearer ${jevApiKey}` } : {};
const proxy = {
  '/v1/systemone': {
    target: jevTarget,
    changeOrigin: true,
    headers: jevHeaders,
  },
  '/jev-health': {
    target: jevTarget,
    changeOrigin: true,
    headers: jevHeaders,
    rewrite: () => '/health',
  },
};
export default defineConfig({
  base: './',
  server: { proxy },
  preview: { proxy },
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    rolldownOptions: {
      input: {
        learn: 'learn.html',
        village: 'index.html',
        driving: 'driving.html',
        mario: 'mario.html',
      },
    },
  },
});
