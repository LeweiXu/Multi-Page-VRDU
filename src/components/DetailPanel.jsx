import React from 'react';
import { getHex } from '../utils/colours';
import { YEAR_MAP } from '../utils/colours';
import './DetailPanel.css';

export default function DetailPanel({ node, edges, nodes, onClose, onSelectNode }) {
  if (!node) return null;

  const year  = YEAR_MAP[node.name] || null;
  const color = getHex(node.type);

  // Incoming edges (who depends on this node)
  const incoming = edges.filter(e => e.target === node.name);
  // Outgoing edges (what this node depends on)
  const outgoing = edges.filter(e => e.source === node.name);

  const nodeByName = name => nodes.find(n => n.name === name);

  return (
    <aside className="detail-panel">
      {/* Header */}
      <div className="dp-header" style={{ borderLeftColor: color }}>
        <div className="dp-meta">
          <span className="dp-tier">T{node.tier}</span>
          {year && <span className="dp-year">{year}</span>}
        </div>
        <h2 className="dp-name">{node.name}</h2>
        <p className="dp-type" style={{ color }}>{node.type}</p>
        <button className="dp-close" onClick={onClose} title="Close">×</button>
      </div>

      <div className="dp-body">
        {node.description && (
          <section className="dp-section">
            <h3 className="dp-section-title">Description</h3>
            <p className="dp-text">{node.description}</p>
          </section>
        )}

        {node.technique && node.technique !== 'N/A' && (
          <section className="dp-section">
            <h3 className="dp-section-title">Key Technique</h3>
            <p className="dp-text">{node.technique}</p>
          </section>
        )}

        {node.architecture && node.architecture !== 'N/A' && (
          <section className="dp-section">
            <h3 className="dp-section-title">Architecture</h3>
            <p className="dp-text">{node.architecture}</p>
          </section>
        )}

        {node.objective && node.objective !== 'N/A' && (
          <section className="dp-section">
            <h3 className="dp-section-title">Training Objective</h3>
            <p className="dp-text">{node.objective}</p>
          </section>
        )}

        {node.bibtex && node.bibtex !== 'N/A' && (
          <section className="dp-section">
            <h3 className="dp-section-title">BibTeX key</h3>
            <code className="dp-bibtex">{node.bibtex}</code>
          </section>
        )}

        {/* Dependencies */}
        {outgoing.length > 0 && (
          <section className="dp-section">
            <h3 className="dp-section-title">Depends on ({outgoing.length})</h3>
            <ul className="dp-links">
              {outgoing.map((e, i) => {
                const target = nodeByName(e.target);
                const tColor = target ? getHex(target.type) : '#5a6a8a';
                return (
                  <li key={i} className="dp-link-item">
                    <button
                      className="dp-link-btn"
                      style={{ borderLeftColor: tColor }}
                      onClick={() => target && onSelectNode(target)}
                    >
                      <span className="dp-link-name">{e.target}</span>
                      {e.rel && <span className="dp-link-rel">{e.rel}</span>}
                    </button>
                  </li>
                );
              })}
            </ul>
          </section>
        )}

        {/* Used by */}
        {incoming.length > 0 && (
          <section className="dp-section">
            <h3 className="dp-section-title">Used by ({incoming.length})</h3>
            <ul className="dp-links">
              {incoming.map((e, i) => {
                const src = nodeByName(e.source);
                const sColor = src ? getHex(src.type) : '#5a6a8a';
                return (
                  <li key={i} className="dp-link-item">
                    <button
                      className="dp-link-btn"
                      style={{ borderLeftColor: sColor }}
                      onClick={() => src && onSelectNode(src)}
                    >
                      <span className="dp-link-name">{e.source}</span>
                      {e.rel && <span className="dp-link-rel">{e.rel}</span>}
                    </button>
                  </li>
                );
              })}
            </ul>
          </section>
        )}
      </div>
    </aside>
  );
}
