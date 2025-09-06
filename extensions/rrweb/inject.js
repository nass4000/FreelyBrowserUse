// rrweb content script injector
// Requires rrweb.min.js to be present in this extension directory.
// Buffers events to window.__rrweb_events for the backend to drain via Selenium.

(function(){
  try {
    if (window.__rrweb_started) return;
    window.__rrweb_events = window.__rrweb_events || [];
    if (typeof rrweb !== 'undefined' && rrweb && rrweb.record) {
      rrweb.record({ emit: function(e){ try { window.__rrweb_events.push(e); } catch(_){} } });
      window.__rrweb_started = true;
    }
  } catch (e) { /* ignore */ }
})();

