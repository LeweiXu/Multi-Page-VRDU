/**
 * colours.js
 * Dynamic category -> color mapping derived from CSV categories.
 */

const DEFAULT_COLOR = '#5a6a8a';
const colorCache = new Map();

function normaliseCategory(category) {
  return String(category || '').trim();
}

function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

function hslToHex(h, s, l) {
  const sat = s / 100;
  const light = l / 100;
  const c = (1 - Math.abs((2 * light) - 1)) * sat;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = light - (c / 2);

  let r = 0;
  let g = 0;
  let b = 0;

  if (h < 60) {
    r = c; g = x; b = 0;
  } else if (h < 120) {
    r = x; g = c; b = 0;
  } else if (h < 180) {
    r = 0; g = c; b = x;
  } else if (h < 240) {
    r = 0; g = x; b = c;
  } else if (h < 300) {
    r = x; g = 0; b = c;
  } else {
    r = c; g = 0; b = x;
  }

  const toHex = n => Math.round((n + m) * 255).toString(16).padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function generateColor(category) {
  const hash = hashString(category);
  const hue = hash % 360;
  const sat = 62;
  const light = 58;
  return hslToHex(hue, sat, light);
}

export function getHex(category) {
  const key = normaliseCategory(category);
  if (!key) return DEFAULT_COLOR;
  if (colorCache.has(key)) return colorCache.get(key);

  const color = generateColor(key);
  colorCache.set(key, color);
  return color;
}

export function getLegendEntries(categories) {
  return (categories || [])
    .map(normaliseCategory)
    .filter(Boolean)
    .map(label => ({
      label,
      color: getHex(label),
    }));
}
