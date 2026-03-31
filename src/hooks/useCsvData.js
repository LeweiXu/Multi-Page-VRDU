/**
 * useCsvData.js
 * Custom hook: fetches vrdu_dependencies.csv, parses it, resolves deps.
 * Returns { nodes, edges, loading, error }
 */

import { useState, useEffect } from 'react';
import { loadNodes } from '../utils/csvParser';
import { resolveAllDeps } from '../utils/softMatch';

export default function useCsvData() {
  const [nodes, setNodes]   = useState([]);
  const [edges, setEdges]   = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]   = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function run() {
      try {
        const raw = await loadNodes();
        if (cancelled) return;

        const resolved = resolveAllDeps(raw);

        // Build edge list: { source: name, target: name, rel: string }
        const edgeList = [];
        resolved.forEach(node => {
          node.resolvedDeps.forEach((d, i) => {
            edgeList.push({
              source: node.name,
              target: d.resolved,
              rawDep: d.raw,
              rel:    node.rels[i] || '',
            });
          });
        });

        setNodes(resolved);
        setEdges(edgeList);
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    run();
    return () => { cancelled = true; };
  }, []);

  return { nodes, edges, loading, error };
}
