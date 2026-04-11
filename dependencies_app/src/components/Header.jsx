import React from 'react';
import './Header.css';

export default function Header({ nodeCount, edgeCount, search, onSearch, filterType, onFilterType, allTypes }) {
  return (
    <header className="header">
      <div className="header-brand">
        <span className="header-logo">◈</span>
        <div>
          <h1 className="header-title">VRDU Dependency Graph</h1>
          <p className="header-sub">Visually Rich Document Understanding · {nodeCount} nodes · {edgeCount} edges</p>
        </div>
      </div>

      <div className="header-controls">
        <div className="search-wrap">
          <span className="search-icon">⌕</span>
          <input
            className="search-input"
            type="text"
            placeholder="Search nodes…"
            value={search}
            onChange={e => onSearch(e.target.value)}
            spellCheck={false}
          />
          {search && (
            <button className="search-clear" onClick={() => onSearch('')} title="Clear">×</button>
          )}
        </div>

        <select
          className="type-filter"
          value={filterType}
          onChange={e => onFilterType(e.target.value)}
        >
          <option value="">All types</option>
          {allTypes.map(t => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </div>
    </header>
  );
}
