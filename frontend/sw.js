// Minimal service worker — exists mainly to satisfy Chrome/Android's PWA
// installability requirement (manifest + registered service worker with a
// fetch handler). No offline caching: this app always needs a live backend,
// so caching API responses would just show stale data.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
self.addEventListener('fetch', () => {}); // passthrough — let the network handle every request
