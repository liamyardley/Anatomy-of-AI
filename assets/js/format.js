// Small helpers for showing numbers and formulas on the page.

// Two decimal places, with a proper minus sign and no "-0.00".
export function fmt(value, places = 2) {
  const rounded = Math.abs(value) < 0.5 * 10 ** -places ? 0 : value;
  return rounded.toFixed(places).replace('-', '−');
}

// The same, for use inside a TeX formula (plain hyphen for minus).
export function texNum(value) {
  return (Math.abs(value) < 0.005 ? 0 : value).toFixed(2);
}

// A share as a whole percentage; "n/a" when there is nothing to measure.
export function pct(value) {
  return value === null || value === undefined ? 'n/a' : `${Math.round(value * 100)}%`;
}

const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
export const prefersReducedMotion = () => reducedMotion.matches;

// Put a value into an element. If it changed, flash the background briefly
// so the eye can follow what moved (skipped under reduced motion).
export function setValue(element, text, flash = true) {
  if (element.textContent === text) return;
  const first = element.textContent === '';
  element.textContent = text;
  if (flash && !first && !reducedMotion.matches && element.animate) {
    element.animate(
      [{ backgroundColor: 'rgba(98, 150, 214, .45)' }, { backgroundColor: 'rgba(98, 150, 214, 0)' }],
      { duration: 700, easing: 'ease-out' },
    );
  }
}

// Render a TeX formula with KaTeX (loaded from a CDN by the page). If KaTeX
// is not available, for example offline, show the formula as plain text.
export function renderTex(element, tex, { display = false } = {}) {
  if (element.dataset.tex === tex) return;
  element.dataset.tex = tex;
  if (window.katex) {
    window.katex.render(tex, element, { displayMode: display, throwOnError: false });
  } else {
    element.textContent = tex
      .replace(/\\begin\{aligned\}|\\end\{aligned\}|\\left|\\right|\\quad|\\qquad|&/g, '')
      .replace(/\\\\/g, ' ')
      .replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, '($1)/($2)')
      .replace(/\\sigma/g, 'σ').replace(/\\lambda/g, 'λ')
      .replace(/\\sum/g, 'Σ').replace(/\\ln/g, 'ln').replace(/\\ge/g, '≥')
      .replace(/\\[a-z]+/gi, '').replace(/[{}]/g, '').replace(/\s+/g, ' ').trim();
  }
}

// Small inline pictures of the two classes, for use in text.
export const MARK = {
  red: '<svg class="mark" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="5" fill="var(--class-red)"/></svg>',
  blue: '<svg class="mark" viewBox="0 0 14 14" aria-hidden="true"><path d="M7 1.5 12.6 11.5H1.4Z" fill="var(--class-blue)"/></svg>',
};
