/**
 * softMatch.js
 * Resolves dependency strings to known node names using progressive
 * soft-matching: exact → normalised-exact → substring → token-overlap.
 *
 * This handles cases like:
 *   "LLaMA 3"  matches "LLaMA 3", "Llama", "LLaMA"
 *   "GPT-3"    matches "GPT-3 / LLMs", "GPT"
 *   "LayoutLMv3" matches "LayoutLMv3 (Document AI)"
 */

/** Remove punctuation, lower-case, collapse spaces. */
function normalise(str) {
  return str
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')   // punctuation → space
    .replace(/\s+/g, ' ')
    .trim();
}

/** Tokenise a normalised string into a Set of words. */
function tokens(str) {
  return new Set(normalise(str).split(' ').filter(w => w.length > 1));
}

/** Jaccard similarity between two token sets. */
function jaccard(a, b) {
  const intersection = new Set([...a].filter(x => b.has(x)));
  const union = new Set([...a, ...b]);
  return union.size === 0 ? 0 : intersection.size / union.size;
}

/**
 * Build a resolver from a list of known node names.
 * Returns a function: depString → resolvedName | null
 */
export function buildResolver(knownNames) {
  const normMap = new Map(
    knownNames.map(n => [normalise(n), n])
  );
  const tokenMap = new Map(
    knownNames.map(n => [n, tokens(n)])
  );

  return function resolve(dep) {
    const raw = dep.trim();
    if (!raw) return null;

    // 1. Exact match
    if (knownNames.includes(raw)) return raw;

    const nd = normalise(raw);

    // 2. Normalised exact
    if (normMap.has(nd)) return normMap.get(nd);

    // 3. Substring — dep normalised is contained in node name, or vice versa
    for (const [norm, name] of normMap) {
      if (norm.includes(nd) || nd.includes(norm)) return name;
    }

    // 4. Token-overlap with Jaccard ≥ 0.35
    const depToks = tokens(raw);
    let best = null;
    let bestScore = 0.34;
    for (const [name, nameToks] of tokenMap) {
      const score = jaccard(depToks, nameToks);
      if (score > bestScore) {
        bestScore = score;
        best = name;
      }
    }
    return best; // null if nothing good enough
  };
}

/**
 * Given an array of node objects (with .name and .deps[]),
 * resolve deps to canonical names and add a .resolvedDeps array.
 */
export function resolveAllDeps(nodes) {
  const knownNames = nodes.map(n => n.name);
  const resolve = buildResolver(knownNames);
  return nodes.map(node => ({
    ...node,
    resolvedDeps: node.deps
      .map(dep => ({ raw: dep, resolved: resolve(dep) }))
      .filter(d => d.resolved !== null),
  }));
}
