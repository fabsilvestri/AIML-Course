/* Make a printed deck look like the deck.
 *
 * reveal.css carries a large `@media print` block guarded by
 * `html:not(.print-pdf)`. It exists to turn a deck into a readable PAPER
 * DOCUMENT rather than a slide replica, and it is aggressive: black headings
 * (`color:#000!important`), 20pt body text, left-aligned everything, links
 * underlined and black.
 *
 * reveal normally adds `print-pdf` to <html> itself when it enters its print
 * view, which switches all of that off. On the vendored 5.2.1 that never
 * happens — see AUTHORING 5.4a — so the paper restyling applied to our
 * exports and the PDFs came out in black with the wrong type sizes.
 *
 * Adding the class permanently is not safe: outside the print block reveal
 * also has `.print-pdf .reveal .slide-background{opacity:1!important}`, which
 * would reveal every slide's background at once ON SCREEN.
 *
 * So it goes on for the duration of the print and comes off afterwards. Chrome
 * fires these events for `--print-to-pdf` as well as for Cmd+P, so the
 * generated handouts and a student's own print get the same result.
 */
(function () {
  var html = document.documentElement;
  var on = function () { html.classList.add('print-pdf'); };
  var off = function () { html.classList.remove('print-pdf'); };

  window.addEventListener('beforeprint', on);
  window.addEventListener('afterprint', off);

  // Safari and older WebKit do not fire the events above; the media query
  // listener is the fallback that does work there.
  if (window.matchMedia) {
    var mq = window.matchMedia('print');
    if (mq.addEventListener) {
      mq.addEventListener('change', function (e) { (e.matches ? on : off)(); });
    }
  }
})();
