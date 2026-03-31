import React from 'react';
import { LEGEND_ENTRIES, TYPE_HEX_MAP } from '../utils/colours';
import './Legend.css';

export default function Legend() {
  return (
    <aside className="legend">
      <p className="legend-title">Node types</p>
      <ul className="legend-list">
        {LEGEND_ENTRIES.map(({ label, varName }) => (
          <li key={varName} className="legend-item">
            <span
              className="legend-dot"
              style={{ background: TYPE_HEX_MAP[varName] }}
            />
            <span className="legend-label">{label}</span>
          </li>
        ))}
      </ul>
    </aside>
  );
}
