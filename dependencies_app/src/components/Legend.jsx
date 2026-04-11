import React from 'react';
import { getLegendEntries } from '../utils/colours';
import './Legend.css';

export default function Legend({ categories }) {
  const entries = getLegendEntries(categories);

  return (
    <aside className="legend">
      <p className="legend-title">Node types</p>
      <ul className="legend-list">
        {entries.map(({ label, color }) => (
          <li key={label} className="legend-item">
            <span
              className="legend-dot"
              style={{ background: color }}
            />
            <span className="legend-label">{label}</span>
          </li>
        ))}
      </ul>
    </aside>
  );
}
