// Boot: formulas, navigation, then every section once its data is loaded.

import { loadJSON, placeholder, resetPalette } from "./util.js";
import { T } from "./i18n.js";
import { initToy } from "./toy.js";
import { initPaper } from "./paper.js";
import { prepareDigits, initDigits } from "./digits.js";
import { prepareForecast, initForecast } from "./forecast.js";

const ASSETS = new URL("../assets/", import.meta.url).href;
const $ = (id) => document.getElementById(id);
const sections = [];

main();

async function main() {
  setupTex();
  setupScrollSpy();
  sections.push(initToy($("viz-toy")));
  sections.push(initPaper($("viz-paper")));

  const digitIds = ["viz-dg-map", "viz-dg-draw", "viz-dg-grid", "viz-kl", "viz-vae-cloud", "viz-vae-map", "viz-beta", "viz-interp"];
  const forecastIds = ["viz-linear", "viz-lowrank", "viz-window", "viz-order", "viz-lab", "viz-fan"];
  for (const id of [...digitIds, ...forecastIds]) $(id).replaceChildren(placeholder(T.loading));

  const digits = Promise.all([loadJSON(ASSETS + "digits.json"), loadJSON(ASSETS + "digit-models.json")])
    .then(([d, m]) => {
      const D = prepareDigits(d, m);
      sections.push(initDigits(D, {
        map: $("viz-dg-map"), draw: $("viz-dg-draw"), grid: $("viz-dg-grid"), kl: $("viz-kl"), cloud: $("viz-vae-cloud"),
        vaeMap: $("viz-vae-map"), beta: $("viz-beta"), interp: $("viz-interp"),
      }));
    })
    .catch((e) => fail(digitIds, e));
  const forecast = loadJSON(ASSETS + "etth1.json")
    .then((j) => {
      const F = prepareForecast(j);
      sections.push(initForecast(F, {
        linear: $("viz-linear"), lowrank: $("viz-lowrank"), window: $("viz-window"), order: $("viz-order"), lab: $("viz-lab"), fan: $("viz-fan"),
      }));
    })
    .catch((e) => fail(forecastIds, e));
  await Promise.all([digits, forecast]);

  let lastWidth = innerWidth;
  let timer = 0;
  addEventListener("resize", () => {
    if (innerWidth === lastWidth) return;
    lastWidth = innerWidth;
    clearTimeout(timer);
    timer = setTimeout(redrawAll, 180);
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    resetPalette();
    redrawAll();
  });
}

function redrawAll() {
  for (const s of sections) s?.redraw?.();
}

function fail(ids, e) {
  console.error(e);
  for (const id of ids) $(id).replaceChildren(placeholder(T.loadError(e.message), true));
}

// KaTeX loads async from the CDN; until then the plain-text fallback inside each .tex span stays visible.
function setupTex() {
  const render = () => {
    for (const el of document.querySelectorAll(".tex[data-tex]")) {
      try {
        window.katex.render(el.dataset.tex, el, { displayMode: !!el.closest(".formula"), throwOnError: false, strict: "ignore" });
      } catch {
        /* keep the fallback text */
      }
    }
  };
  if (window.katex) render();
  else document.getElementById("katex-js")?.addEventListener("load", render);
}

function setupScrollSpy() {
  const links = [...document.querySelectorAll("#pipeline a")];
  const targets = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  const onScroll = () => {
    const y = scrollY + 120;
    let cur = null;
    for (const t of targets) if (t.offsetTop <= y) cur = t;
    for (const a of links) a.classList.toggle("active", cur && a.getAttribute("href") === `#${cur.id}`);
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();
}
