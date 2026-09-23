/* Course material appears on the day of its lecture, not before.
 *
 * No lecture's slides, notebook or notes are linked in index.html. Each card
 * carries only the moment it may be unveiled (data-reveal, an absolute instant
 * with its offset, so it means the same to a reader in any timezone) and the
 * kinds of material that lecture owns (data-material). This builds the buttons
 * from the lecture number, and only once that instant has passed.
 *
 * What this is and is not: it keeps the material off the page until the class,
 * so a student reading ahead sees a calendar rather than the course. It is not
 * access control. The files sit in a public repository at predictable paths
 * and anyone who wants them before the lecture can have them; gating them
 * properly would mean not publishing them, which is a different decision.
 *
 * The page is re-checked when the soonest future reveal falls due, so a tab
 * left open across 11:30 fills in rather than showing a locked card until
 * someone reloads.
 *
 * The clock is the SERVER's, not the reader's. A gate that compares against
 * the reader's own clock opens for anyone willing to set their laptop forward
 * a month, which is a strange thing to make a student discover. The Date
 * header on a HEAD of this page is the host's clock; the difference between it
 * and the local clock is kept and applied to every comparison, so moving the
 * laptop's clock moves nothing. If that request fails -- offline, a proxy that
 * eats the header -- the local clock is used and the gate still works; it is
 * the cheat that stops being caught, not the reveal.
 */
(function () {
  "use strict";

  var COLAB = "https://colab.research.google.com/github/fabsilvestri/" +
              "AIML-Course/blob/main/notebooks/lecture-";

  // Must stay in step with MATERIAL in tools/make_site.py, which is what
  // decides the kinds written into data-material in the first place.
  var KINDS = {
    slides:   { cls: "btn",           label: "Slides",
                href: function (nn) { return "slides/lecture-" + nn + ".html"; } },
    pdf:      { cls: "btn btn-pdf",   label: "PDF",
                href: function (nn) { return "slides/pdf/lecture-" + nn + ".pdf"; } },
    notebook: { cls: "btn btn-colab", label: "Notebook",
                href: function (nn) { return COLAB + nn + ".ipynb"; } },
    notes:    { cls: "btn btn-notes", label: "Notes (PDF)",
                href: function (nn) { return "notes/lecture-" + nn + ".pdf"; } }
  };
  var ORDER = ["slides", "pdf", "notebook", "notes"];

  var cards = Array.prototype.slice.call(
    document.querySelectorAll(".lecture[data-reveal]"));
  if (!cards.length) return;

  // Milliseconds to add to the local clock to get the host's. Zero until the
  // HEAD below says otherwise, and zero forever if it cannot.
  var skew = 0;
  function now() { return new Date(Date.now() + skew); }

  function syncClock() {
    // A bounded wait: a request that never settles must not leave every card
    // shut for the whole lecture. Four seconds, then give up and use the local
    // clock -- which is the honest reader's clock, and correct.
    return new Promise(function (resolve) {
      var done = false;
      function finish() { if (!done) { done = true; resolve(); } }
      setTimeout(finish, 4000);
      try {
        // no-store, and a unique query, so a cached response cannot hand back
        // a Date header from this morning.
        fetch(location.pathname + "?clock=" + Date.now(),
              { method: "HEAD", cache: "no-store" })
          .then(function (res) {
            var header = res.headers.get("Date");
            var at = header ? Date.parse(header) : NaN;
            if (!isNaN(at)) skew = at - Date.now();
            finish();
          })
          .catch(finish);
      } catch (e) {
        finish();
      }
    });
  }

  function unveil(card) {
    var nn = card.getAttribute("data-n");
    var want = (card.getAttribute("data-material") || "").split(",");
    var links = card.querySelector(".links");
    if (!nn || !links) return;

    var frag = document.createDocumentFragment();
    ORDER.forEach(function (kind) {
      if (want.indexOf(kind) === -1) return;
      var spec = KINDS[kind];
      var a = document.createElement("a");
      a.className = spec.cls;
      a.href = spec.href(nn);
      a.textContent = spec.label;
      frag.appendChild(a);
    });
    // A lecture whose data-material named nothing we know how to build would
    // otherwise lose its chip and show an empty row, which reads as a bug
    // rather than as a lecture with no material. Leave the card alone.
    if (!frag.childNodes.length) return;

    links.textContent = "";
    links.appendChild(frag);
    card.setAttribute("data-revealed", "true");
  }

  function markNext(now) {
    var rows = document.querySelectorAll(".calendar tr[data-date]");
    var line = document.querySelector(".cal-next");
    var i, when, row = null;
    for (i = 0; i < rows.length; i++) {
      // A lecture counts as still to come until the end of its own day, so the
      // row stays marked while the class it names is being taught.
      when = new Date(rows[i].getAttribute("data-date") + "T23:59:59");
      rows[i].classList.remove("is-next");
      if (!row && when >= now) row = rows[i];
    }
    if (!row || !line) return;
    row.classList.add("is-next");
    var n = row.querySelector(".cal-n");
    var day = row.querySelector(".cal-when");
    line.textContent = "Next · lecture " + (n ? n.textContent : "") +
                       ", " + (day ? day.textContent : "");
    line.hidden = false;
  }

  function sweep() {
    var current = now();
    var soonest = null;

    cards.forEach(function (card) {
      var at = new Date(card.getAttribute("data-reveal"));
      if (isNaN(at)) return;              // an unparseable date stays locked
      if (at <= current) {
        if (!card.hasAttribute("data-revealed")) unveil(card);
      } else if (soonest === null || at < soonest) {
        soonest = at;
      }
    });

    markNext(current);

    // setTimeout overflows past about 24 days and fires immediately, which
    // would spin. Wake at the next reveal, or in a day, whichever is sooner.
    if (soonest !== null) {
      var DAY = 86400000;
      setTimeout(sweep, Math.min(Math.max(soonest - current + 1000, 1000), DAY));
    }
  }

  // Nothing is unveiled before the clock question is settled. The cards ship
  // shut, so waiting shows a correct page rather than a flash of material.
  syncClock().then(sweep);
})();
