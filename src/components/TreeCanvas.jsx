import React, { useRef, useEffect, useCallback, useState } from 'react';
import * as d3 from 'd3';
import { getHex, YEAR_MAP } from '../utils/colours';
import { NODE_W_EXPORT as NODE_W, NODE_H_EXPORT as NODE_H } from '../hooks/useTreeLayout';
import './TreeCanvas.css';

const TIER_LABELS = ['T0 · Foundations', 'T1 · Pre-Training', 'T2 · Fine-Tuning', 'T3 · LLMs', 'T4 · Tools', 'T5 · Multi-Page VRDU'];
const TIER_BG_ALPHA = [0.04, 0.03, 0.025, 0.02, 0.025, 0.05];

export default function TreeCanvas({
  layoutNodes,
  edges,
  svgWidth,
  svgHeight,
  selectedNode,
  onSelectNode,
  search,
  filterType,
}) {
  const svgRef    = useRef(null);
  const groupRef  = useRef(null);
  const zoomRef   = useRef(null);
  const [transform, setTransform] = useState({ k: 1, x: 0, y: 0 });

  // ── D3 zoom setup ────────────────────────────────────────────────
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);

    const zoom = d3.zoom()
      .scaleExtent([0.15, 3])
      .on('zoom', event => {
        const t = event.transform;
        setTransform({ k: t.k, x: t.x, y: t.y });
        if (groupRef.current) {
          d3.select(groupRef.current).attr('transform', `translate(${t.x},${t.y}) scale(${t.k})`);
        }
      });

    svg.call(zoom);
    zoomRef.current = zoom;

    // Initial fit
    const containerW = svgRef.current.parentElement?.clientWidth || 1200;
    const containerH = svgRef.current.parentElement?.clientHeight || 700;
    const scale = Math.min(containerW / (svgWidth + 40), containerH / (svgHeight + 40), 1);
    const tx = (containerW - svgWidth * scale) / 2;
    const ty = (containerH - svgHeight * scale) / 2;
    const initTransform = d3.zoomIdentity.translate(tx, ty).scale(scale);
    svg.call(zoom.transform, initTransform);

    return () => svg.on('.zoom', null);
  // eslint-disable-next-line
  }, [svgWidth, svgHeight]);

  // Zoom controls
  const zoomIn  = useCallback(() => { if (!svgRef.current || !zoomRef.current) return; d3.select(svgRef.current).transition().call(zoomRef.current.scaleBy, 1.3); }, []);
  const zoomOut = useCallback(() => { if (!svgRef.current || !zoomRef.current) return; d3.select(svgRef.current).transition().call(zoomRef.current.scaleBy, 0.77); }, []);
  const zoomFit = useCallback(() => {
    if (!svgRef.current || !zoomRef.current) return;
    const containerW = svgRef.current.parentElement?.clientWidth || 1200;
    const containerH = svgRef.current.parentElement?.clientHeight || 700;
    const scale = Math.min(containerW / (svgWidth + 40), containerH / (svgHeight + 40), 1);
    const tx = (containerW - svgWidth * scale) / 2;
    const ty = (containerH - svgHeight * scale) / 2;
    d3.select(svgRef.current).transition().duration(400).call(
      zoomRef.current.transform, d3.zoomIdentity.translate(tx, ty).scale(scale)
    );
  }, [svgWidth, svgHeight]);

  // ── Derived visibility ────────────────────────────────────────────
  const nodeMap = {};
  layoutNodes.forEach(n => { nodeMap[n.name] = n; });

  const isHighlighted = useCallback(node => {
    if (!selectedNode) return false;
    if (node.name === selectedNode.name) return true;
    const isDep = edges.some(e => e.source === selectedNode.name && e.target === node.name);
    const isUser = edges.some(e => e.target === selectedNode.name && e.source === node.name);
    return isDep || isUser;
  }, [selectedNode, edges]);

  const matchesSearch = useCallback(node => {
    if (!search) return true;
    const q = search.toLowerCase();
    return (
      node.name.toLowerCase().includes(q) ||
      node.type.toLowerCase().includes(q) ||
      node.description.toLowerCase().includes(q)
    );
  }, [search]);

  const matchesFilter = useCallback(node => {
    if (!filterType) return true;
    return node.type === filterType;
  }, [filterType]);

  const nodeOpacity = useCallback(node => {
    if (selectedNode) return isHighlighted(node) ? 1 : 0.18;
    const ms = matchesSearch(node);
    const mf = matchesFilter(node);
    if (!ms || !mf) return 0.15;
    return 1;
  }, [selectedNode, isHighlighted, matchesSearch, matchesFilter]);

  // ── Tier stripe backgrounds ───────────────────────────────────────
  // Group layoutNodes by tier to find y extents
  const tierExtents = {};
  layoutNodes.forEach(n => {
    if (!tierExtents[n.tier]) tierExtents[n.tier] = { minY: n.y, maxY: n.y };
    tierExtents[n.tier].minY = Math.min(tierExtents[n.tier].minY, n.y);
    tierExtents[n.tier].maxY = Math.max(tierExtents[n.tier].maxY, n.y);
  });

  // ── Edge path ─────────────────────────────────────────────────────
  function edgePath(src, tgt) {
    const x1 = src.x + NODE_W / 2;
    const y1 = src.y + NODE_H;
    const x2 = tgt.x + NODE_W / 2;
    const y2 = tgt.y;
    const cy = (y1 + y2) / 2;
    return `M${x1},${y1} C${x1},${cy} ${x2},${cy} ${x2},${y2}`;
  }

  // ── Render ────────────────────────────────────────────────────────
  return (
    <div className="canvas-wrap">
      {/* Zoom controls */}
      <div className="zoom-controls">
        <button className="zoom-btn" onClick={zoomIn}  title="Zoom in">+</button>
        <button className="zoom-btn" onClick={zoomFit} title="Fit">⊡</button>
        <button className="zoom-btn" onClick={zoomOut} title="Zoom out">−</button>
      </div>

      {/* Zoom level badge */}
      <div className="zoom-badge">{Math.round(transform.k * 100)}%</div>

      <svg
        ref={svgRef}
        className="tree-svg"
        width="100%"
        height="100%"
        style={{ cursor: 'grab' }}
      >
        <defs>
          <marker
            id="arrowhead"
            markerWidth="6" markerHeight="6"
            refX="5" refY="3"
            orient="auto"
          >
            <path d="M0,0 L0,6 L6,3 z" fill="rgba(255,255,255,0.25)" />
          </marker>
          <marker
            id="arrowhead-selected"
            markerWidth="6" markerHeight="6"
            refX="5" refY="3"
            orient="auto"
          >
            <path d="M0,0 L0,6 L6,3 z" fill="#4f9cf9" />
          </marker>
          <filter id="glow">
            <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <g ref={groupRef}>
          {/* Tier stripes */}
          {Object.entries(tierExtents).map(([tier, ext]) => {
            const t = parseInt(tier, 10);
            const alpha = TIER_BG_ALPHA[t] || 0.03;
            const stripeY = ext.minY - 28;
            const stripeH = ext.maxY - ext.minY + NODE_H + 52;
            return (
              <g key={tier}>
                <rect
                  x={0}
                  y={stripeY}
                  width={svgWidth}
                  height={stripeH}
                  fill={`rgba(255,255,255,${alpha})`}
                  rx={8}
                />
                <text
                  x={12}
                  y={stripeY + 16}
                  fontSize={10}
                  fill="rgba(255,255,255,0.18)"
                  fontFamily="var(--font-display)"
                  fontWeight={700}
                  letterSpacing="0.08em"
                  textTransform="uppercase"
                >
                  {TIER_LABELS[t] || `T${t}`}
                </text>
              </g>
            );
          })}

          {/* Edges */}
          {edges.map((edge, i) => {
            const src = nodeMap[edge.source];
            const tgt = nodeMap[edge.target];
            if (!src || !tgt) return null;

            const isActive = selectedNode && (
              edge.source === selectedNode.name || edge.target === selectedNode.name
            );
            const isIncoming = selectedNode && edge.target === selectedNode.name;

            if (selectedNode && !isActive) return null;

            return (
              <path
                key={i}
                d={edgePath(src, tgt)}
                fill="none"
                stroke={
                  isActive
                    ? (isIncoming ? '#e8c45a' : '#4f9cf9')
                    : 'rgba(255,255,255,0.1)'
                }
                strokeWidth={isActive ? 1.8 : 0.8}
                strokeDasharray={isActive ? 'none' : '3,3'}
                markerEnd={isActive ? 'url(#arrowhead-selected)' : 'url(#arrowhead)'}
                opacity={isActive ? 1 : 0.5}
              />
            );
          })}

          {/* All edges (dimmed) when no selection */}
          {!selectedNode && edges.map((edge, i) => {
            const src = nodeMap[edge.source];
            const tgt = nodeMap[edge.target];
            if (!src || !tgt) return null;
            return (
              <path
                key={`dim-${i}`}
                d={edgePath(src, tgt)}
                fill="none"
                stroke="rgba(255,255,255,0.07)"
                strokeWidth={0.7}
                markerEnd="url(#arrowhead)"
              />
            );
          })}

          {/* Nodes */}
          {layoutNodes.map(node => {
            const color  = getHex(node.type);
            const year   = YEAR_MAP[node.name];
            const opacity = nodeOpacity(node);
            const isSelected = selectedNode?.name === node.name;
            const isConnected = selectedNode && isHighlighted(node) && !isSelected;

            return (
              <g
                key={node.name}
                transform={`translate(${node.x},${node.y})`}
                opacity={opacity}
                style={{ cursor: 'pointer', transition: 'opacity 0.2s' }}
                onClick={() => onSelectNode(isSelected ? null : node)}
              >
                {/* Glow on selected */}
                {isSelected && (
                  <rect
                    x={-3} y={-3}
                    width={NODE_W + 6} height={NODE_H + 6}
                    rx={9}
                    fill="none"
                    stroke={color}
                    strokeWidth={1.5}
                    opacity={0.5}
                    filter="url(#glow)"
                  />
                )}

                {/* Main box */}
                <rect
                  x={0} y={0}
                  width={NODE_W} height={NODE_H}
                  rx={6}
                  fill={isSelected ? `${color}22` : '#12161f'}
                  stroke={isSelected ? color : (isConnected ? `${color}88` : 'rgba(255,255,255,0.09)')}
                  strokeWidth={isSelected ? 1.5 : (isConnected ? 1.2 : 0.8)}
                />

                {/* Colour tab */}
                <rect
                  x={0} y={0}
                  width={3} height={NODE_H}
                  rx={3}
                  fill={color}
                  opacity={isSelected ? 1 : 0.7}
                />

                {/* Year tag */}
                {year && (
                  <text
                    x={NODE_W - 5} y={10}
                    textAnchor="end"
                    fontSize={8}
                    fill="rgba(255,255,255,0.25)"
                    fontFamily="var(--font-mono)"
                  >
                    {year}
                  </text>
                )}

                {/* Node name */}
                <foreignObject x={7} y={0} width={NODE_W - 14} height={NODE_H}>
                  <div
                    xmlns="http://www.w3.org/1999/xhtml"
                    style={{
                      height: `${NODE_H}px`,
                      display: 'flex',
                      alignItems: 'center',
                      paddingTop: year ? '6px' : '0',
                    }}
                  >
                    <span
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: node.name.length > 18 ? '9px' : node.name.length > 12 ? '10px' : '11px',
                        color: isSelected ? '#e2e8f8' : 'rgba(226,232,248,0.82)',
                        lineHeight: 1.25,
                        wordBreak: 'break-word',
                        overflow: 'hidden',
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                      }}
                    >
                      {node.name}
                    </span>
                  </div>
                </foreignObject>
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}
