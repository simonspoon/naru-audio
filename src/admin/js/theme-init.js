// Sets <html data-theme> before first paint so a non-dark theme does not
// flash dark while the module scripts load. A classic script because the CSP
// (script-src 'self') allows no inline one; app.mjs's applyTheme() owns the
// theme after this, with the same key and rules.
(function () {
  var ids = ['dark', 'light', 'contrast', 'crimson'];
  var v = null;
  try {
    v = localStorage.getItem('naru-audio-theme');
  } catch (e) {
    // storage blocked: follow the system
  }
  if (ids.indexOf(v) < 0) v = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  document.documentElement.dataset.theme = v;
})();
