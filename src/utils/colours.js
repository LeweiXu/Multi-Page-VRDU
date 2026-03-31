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

export const YEAR_MAP = {
  'BM25': 1994, 'Transformer': 2017, 'BERT': 2018, 'ViT': 2020,
  'T5': 2019, 'DiT (Document Image Transformer)': 2022, 'BEiT': 2021,
  'ColBERT': 2020, 'CLIP': 2021, 'Pix2Struct': 2022,
  'SigLIP-SO400m': 2023, 'DPR': 2020, 'LLaMA 3': 2024,
  'LayoutLMv3': 2022, 'LLaVA': 2023, 'RAG': 2020, 'ColPali': 2024,
  'InternViT-6B': 2024, 'Qwen2-VL': 2024, 'InternVL2': 2024,
  'PaliGemma': 2024, 'DocFormerv2': 2023, 'CoT (Chain-of-Thought)': 2022,
  'MinerU': 2024, 'DocLayNet': 2022, 'Adapter (Houlsby 2019)': 2019,
  'MLP Projector': 2023, 'LoRA': 2021, 'ReAct': 2022,
  'ViDoRAG': 2025, 'Doc-React': 2025, 'Doc-Agent': 2025,
  'MDocAgent': 2025, 'M3DocRAG': 2024, 'VDocRAG': 2025,
  'MoLoRAG': 2025, 'AVIR': 2026, 'KGP': 2024, 'LayTokenLLM': 2025,
  'CREAM': 2024, 'MHier-RAG': 2025, 'Self-Attention Scoring': 2024,
  'InstructDr': 2024, 'PDF-WuKong': 2024, 'Texthawk2': 2024,
  'Leopard': 2024, 'mPLUG-DocOwl2': 2024, 'LayoutLLM': 2024,
  'Docopilot': 2025, 'Hi-VT5': 2022, 'GRAM': 2024,
  'Arctic-TILT': 2024, 'DocLLM': 2024, 'DocSLM': 2026,
  'DocLayLLM': 2025,
};
