/* Marks the nav link for the section you are currently reading.
 *
 * A scroll handler rather than IntersectionObserver. IO is the tidier tool,
 * but its rootMargin only accepts pixels or percent -- a rem value throws and
 * leaves you with a scroll-spy that silently never spies -- and I could not
 * verify it firing in the harness I had. This does the same job with one
 * comparison per section per frame, on a page of eight sections, and it can
 * be checked by reading the DOM after a scroll.
 *
 * Reads are batched into a rAF callback so a fast scroll costs one layout
 * pass rather than one per event.
 *
 * State is written as aria-current, so the accessible state and the visible
 * state are the same fact rather than two that can drift.
 */
(function () {
  "use strict";

  var links = Array.prototype.slice.call(
    document.querySelectorAll('.site-nav a.nav-link[href^="#"]'));
  if (!links.length) return;

  var sections = [];
  links.forEach(function (a) {
    var el = document.getElementById(a.getAttribute("href").slice(1));
    if (el) sections.push({ el: el, link: a });
  });
  if (!sections.length) return;

  var current = null;
  function mark(link) {
    if (link === current) return;
    current = link;
    links.forEach(function (a) {
      if (a === link) a.setAttribute("aria-current", "true");
      else a.removeAttribute("aria-current");
    });
  }

  function navHeight() {
    var bar = document.querySelector(".site-nav");
    return bar ? bar.getBoundingClientRect().height : 50;
  }

  function update() {
    // The section being read is the last one whose top has passed just under
    // the sticky bar. A small extra margin stops the highlight flickering
    // between two sections at the exact boundary.
    var line = navHeight() + 8;
    var found = sections[0];
    for (var i = 0; i < sections.length; i++) {
      if (sections[i].el.getBoundingClientRect().top <= line) found = sections[i];
      else break;
    }
    // At the very bottom the last section may be too short to reach the line;
    // if the page is scrolled to the end, it is the one being read.
    if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 2) {
      found = sections[sections.length - 1];
    }
    mark(found.link);
  }

  var ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    window.requestAnimationFrame(function () { ticking = false; update(); });
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll, { passive: true });
  update();
})();
