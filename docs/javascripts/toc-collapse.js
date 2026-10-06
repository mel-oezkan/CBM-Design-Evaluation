// Collapsible table of contents for long reference pages. A page opts in with front matter
// `toc_collapse: <depth>`: entries at that depth or deeper start collapsed (1 = top-level sections,
// 2 = their children, e.g. the classes under a module). Every entry with children gets a toggle,
// and the entry of the section being read opens on its own while you scroll.
(function () {
  function setup() {
    var meta = document.querySelector('meta[name="cbm-toc-collapse"]');
    if (!meta) return;
    var depth = parseInt(meta.content, 10) || 1;
    document.querySelectorAll(".md-nav--secondary > .md-nav__list").forEach(function (root) {
      if (root.dataset.cbmToc) return;
      root.dataset.cbmToc = "1";
      init(root, depth);
    });
  }

  function init(root, depth) {
    root.querySelectorAll("li.md-nav__item").forEach(function (li) {
      var child = li.querySelector(":scope > nav");
      var link = li.querySelector(":scope > a.md-nav__link");
      if (!child || !link) return;
      var level = 1;
      for (var p = li.parentElement; p && p !== root; p = p.parentElement) {
        if (p.tagName === "LI") level++;
      }
      var button = document.createElement("button");
      button.type = "button";
      button.className = "cbm-toc-toggle";
      button.setAttribute("aria-label", "Toggle " + link.textContent.trim());
      button.addEventListener("click", function () {
        delete li.dataset.cbmAuto;
        toggle(li, li.classList.contains("cbm-toc-collapsed"));
      });
      li.classList.add("cbm-toc-parent");
      li.insertBefore(button, link);
      toggle(li, level < depth);
    });

    // Follow the scroll position: open the ancestors of the active entry, and close again the
    // ones that were only opened that way. Entries the reader toggled stay as they are.
    function follow() {
      var active = root.querySelector(".md-nav__link--active");
      var keep = new Set();
      for (var p = active && active.parentElement; p && p !== root; p = p.parentElement) {
        if (p.classList.contains("cbm-toc-parent")) keep.add(p);
      }
      root.querySelectorAll("li[data-cbm-auto]").forEach(function (li) {
        if (!keep.has(li)) { delete li.dataset.cbmAuto; toggle(li, false); }
      });
      keep.forEach(function (li) {
        if (li.classList.contains("cbm-toc-collapsed")) { li.dataset.cbmAuto = "1"; toggle(li, true); }
      });
    }
    new MutationObserver(follow).observe(root, { subtree: true, attributes: true, attributeFilter: ["class"] });
    follow();
  }

  function toggle(li, open) {
    li.classList.toggle("cbm-toc-collapsed", !open);
    li.querySelector(":scope > .cbm-toc-toggle").setAttribute("aria-expanded", String(open));
  }

  // Material's instant navigation re-renders the page through `document$`; without it, run once.
  if (window.document$) window.document$.subscribe(setup);
  else document.addEventListener("DOMContentLoaded", setup);
})();
