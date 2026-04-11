/**
 * useTreeLayout.js
 * Computes (x, y) positions for every node in a tier-row grid.
 *
 * Layout rules:
 *  - Tiers 0-4: one row per tier, nodes spaced horizontally, centred.
 *  - Tier 5: wraps at TIER5_COLS per row (default 10), creating multiple rows.
 *
 * Returns { layoutNodes, svgWidth, svgHeight }
 * where layoutNodes[i] = { ...node, x, y }
 */

import { useMemo } from 'react';

const NODE_W     = 130;   // node box width  (px)
const NODE_H     = 42;    // node box height (px)
const H_GAP      = 22;    // horizontal gap between nodes
const V_GAP      = 72;    // vertical gap between tier rows
const PAD_TOP    = 60;    // top padding
const PAD_X      = 80;    // left/right padding
const TIER5_COLS = 10;    // max nodes per row in tier 5

export const NODE_W_EXPORT = NODE_W;
export const NODE_H_EXPORT = NODE_H;

export default function useTreeLayout(nodes) {
  return useMemo(() => {
    if (!nodes.length) return { layoutNodes: [], svgWidth: 800, svgHeight: 400 };

    // Group by tier
    const byTier = {};
    nodes.forEach(n => {
      if (!byTier[n.tier]) byTier[n.tier] = [];
      byTier[n.tier].push(n);
    });

    const tiers = Object.keys(byTier).map(Number).sort((a, b) => a - b);

    // Calculate rows: for tier < 5 → 1 row; for tier 5 → ceil(count/TIER5_COLS) rows
    // rowYs maps (tier, rowIndex) → y
    const layoutNodes = [];
    let currentY = PAD_TOP;

    // We need the max row width to determine SVG width
    let maxRowPixelWidth = 0;

    // Process each tier in order
    tiers.forEach(tier => {
      const tierNodes = byTier[tier];

      if (tier < 5) {
        // Single row
        const rowWidth = tierNodes.length * (NODE_W + H_GAP) - H_GAP;
        if (rowWidth > maxRowPixelWidth) maxRowPixelWidth = rowWidth;

        tierNodes.forEach((node, i) => {
          layoutNodes.push({ ...node, _rowY: currentY, _col: i, _rowTotal: tierNodes.length });
        });
        currentY += NODE_H + V_GAP;
      } else {
        // Tier 5: wrap into chunks of TIER5_COLS
        const chunks = [];
        for (let i = 0; i < tierNodes.length; i += TIER5_COLS) {
          chunks.push(tierNodes.slice(i, i + TIER5_COLS));
        }
        chunks.forEach(chunk => {
          const rowWidth = chunk.length * (NODE_W + H_GAP) - H_GAP;
          if (rowWidth > maxRowPixelWidth) maxRowPixelWidth = rowWidth;

          chunk.forEach((node, i) => {
            layoutNodes.push({ ...node, _rowY: currentY, _col: i, _rowTotal: chunk.length });
          });
          currentY += NODE_H + V_GAP;
        });
      }
    });

    const svgWidth  = maxRowPixelWidth + PAD_X * 2;
    const svgHeight = currentY + NODE_H + PAD_TOP;

    // Now assign x by centring each row in svgWidth
    const result = layoutNodes.map(n => {
      const rowWidth = n._rowTotal * (NODE_W + H_GAP) - H_GAP;
      const startX = (svgWidth - rowWidth) / 2;
      const x = startX + n._col * (NODE_W + H_GAP);
      const y = n._rowY;
      return { ...n, x, y };
    });

    return { layoutNodes: result, svgWidth, svgHeight };
  }, [nodes]);
}

export { TIER5_COLS, NODE_W, NODE_H, V_GAP, H_GAP, PAD_TOP, PAD_X };
