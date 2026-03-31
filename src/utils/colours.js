/**
 * colours.js
 * Maps node `type` strings to CSS variable names and hex values.
 */

// CSS variable names defined in global.css
export const TYPE_VAR_MAP = {
  // Tier 0-1 foundations
  'Foundation Architecture':           '--cat-foundation-nlp',
  'Pretrained Language Model':         '--cat-foundation-nlp',
  'Open-Weight Large Language Model':  '--cat-foundation-nlp',
  'Dense Retrieval Model':             '--cat-retrieval',
  'Retrieval Algorithm':               '--cat-retrieval',
  'Retrieval-Augmented Generation Framework': '--cat-retrieval',

  // Vision
  'Vision Foundation Model':           '--cat-foundation-vision',
  'Document Vision Foundation Model':  '--cat-foundation-vision',
  'Visual Document Retrieval Model':   '--cat-foundation-vision',

  // VLM / Multimodal
  'Vision-Language Model':             '--cat-vlm',
  'Multimodal Large Language Model':   '--cat-vlm',
  'Multimodal Document Pre-trained Model': '--cat-vlm',

  // Agents & prompting
  'Agentic Prompting Framework':       '--cat-agent',
  'Prompting Framework':               '--cat-agent',

  // PEFT / components
  'Parameter-Efficient Fine-Tuning (PEFT) Method': '--cat-peft',
  'Architectural Component / Connector': '--cat-peft',

  // Tools / parsing
  'Document Parsing Pipeline':         '--cat-tool',
  'Document Layout Analysis Dataset':  '--cat-tool',

  // Target models
  'Multi-Page DocVQA Model':           '--cat-docvqa',
};

// Hex fallback palette (must match CSS vars above)
export const TYPE_HEX_MAP = {
  '--cat-foundation-nlp':    '#e07b5a',
  '--cat-foundation-vision': '#5a8fe0',
  '--cat-vlm':               '#b87de8',
  '--cat-retrieval':         '#5ac8a0',
  '--cat-agent':             '#e8c45a',
  '--cat-peft':              '#e87a5a',
  '--cat-dataset':           '#7a8fb0',
  '--cat-tool':              '#60b8b0',
  '--cat-docvqa':            '#e05a8f',
  '--cat-default':           '#5a6a8a',
};

export function getCSSVar(type) {
  return TYPE_VAR_MAP[type] || '--cat-default';
}

export function getHex(type) {
  const varName = getCSSVar(type);
  return TYPE_HEX_MAP[varName] || TYPE_HEX_MAP['--cat-default'];
}

// Distinct groups for the legend
export const LEGEND_ENTRIES = [
  { label: 'NLP Foundation',    varName: '--cat-foundation-nlp' },
  { label: 'Vision Foundation', varName: '--cat-foundation-vision' },
  { label: 'VLM / Multimodal', varName: '--cat-vlm' },
  { label: 'Retrieval / RAG',   varName: '--cat-retrieval' },
  { label: 'Agentic / Prompting', varName: '--cat-agent' },
  { label: 'PEFT / Component',  varName: '--cat-peft' },
  { label: 'Tool / Parser',     varName: '--cat-tool' },
  { label: 'DocVQA Model',      varName: '--cat-docvqa' },
];

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
