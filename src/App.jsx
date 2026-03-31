import React, { useState, useMemo } from 'react';
import './App.css';
import useCsvData from './hooks/useCsvData';
import useTreeLayout from './hooks/useTreeLayout';
import Header from './components/Header';
import TreeCanvas from './components/TreeCanvas';
import DetailPanel from './components/DetailPanel';
import Legend from './components/Legend';

export default function App() {
  const { nodes, edges, loading, error } = useCsvData();
  const { layoutNodes, svgWidth, svgHeight } = useTreeLayout(nodes);

  const [selectedNode, setSelectedNode] = useState(null);
  const [search, setSearch] = useState('');
  const [filterType, setFilterType] = useState('');

  // All unique types for the filter dropdown
  const allTypes = useMemo(() => {
    const s = new Set(nodes.map(n => n.type).filter(Boolean));
    return [...s].sort();
  }, [nodes]);

  function handleSelectNode(node) {
    setSelectedNode(node || null);
  }

  if (loading) {
    return (
      <div className="splash">
        <div className="splash-spinner" />
        <p className="splash-text">Loading dependency graph…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="splash splash--error">
        <span className="splash-icon">⚠</span>
        <p className="splash-text">Failed to load CSV</p>
        <p className="splash-detail">{error}</p>
        <p className="splash-hint">
          Make sure <code>vrdu_dependencies.csv</code> is present in the{' '}
          <code>public/</code> folder.
        </p>
      </div>
    );
  }

  const panelOpen = !!selectedNode;

  return (
    <div className="app-root">
      <Header
        nodeCount={nodes.length}
        edgeCount={edges.length}
        search={search}
        onSearch={setSearch}
        filterType={filterType}
        onFilterType={setFilterType}
        allTypes={allTypes}
      />

      <main className={`app-main ${panelOpen ? 'panel-open' : ''}`}>
        <TreeCanvas
          layoutNodes={layoutNodes}
          edges={edges}
          svgWidth={svgWidth}
          svgHeight={svgHeight}
          selectedNode={selectedNode}
          onSelectNode={handleSelectNode}
          search={search}
          filterType={filterType}
        />
      </main>

      {panelOpen && (
        <DetailPanel
          node={selectedNode}
          edges={edges}
          nodes={nodes}
          onClose={() => setSelectedNode(null)}
          onSelectNode={handleSelectNode}
        />
      )}

      <Legend />
    </div>
  );
}
