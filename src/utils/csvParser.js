/**
 * csvParser.js
 * Fetches and parses vrdu_dependencies.csv from the public/ folder.
 * Returns an array of node objects with normalised fields.
 */

const CSV_PATH = process.env.PUBLIC_URL + '/vrdu_dependencies.csv';

/**
 * Minimal CSV parser that handles quoted fields with commas and newlines.
 * Returns { headers: string[], rows: string[][] }
 */
function parseCSV(text) {
  const rows = [];
  let currentRow = [];
  let currentField = '';
  let inQuotes = false;

  // Normalise line endings
  const chars = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

  for (let i = 0; i < chars.length; i++) {
    const ch = chars[i];
    const next = chars[i + 1];

    if (inQuotes) {
      if (ch === '"' && next === '"') {
        // Escaped quote
        currentField += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        currentField += ch;
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
      } else if (ch === ',') {
        currentRow.push(currentField);
        currentField = '';
      } else if (ch === '\n') {
        currentRow.push(currentField);
        currentField = '';
        rows.push(currentRow);
        currentRow = [];
      } else {
        currentField += ch;
      }
    }
  }

  // Final field / row
  if (currentField || currentRow.length > 0) {
    currentRow.push(currentField);
    rows.push(currentRow);
  }

  // Filter completely empty rows
  const cleaned = rows.filter(r => r.some(f => f.trim() !== ''));
  const headers = cleaned[0];
  const dataRows = cleaned.slice(1);
  return { headers, rows: dataRows };
}

/**
 * Fetch and parse the CSV into node objects.
 */
export async function loadNodes() {
  const res = await fetch(CSV_PATH);
  if (!res.ok) throw new Error(`Failed to load CSV: ${res.status} ${res.statusText}`);
  const text = await res.text();
  const { headers, rows } = parseCSV(text);

  return rows
    .map(row => {
      const obj = {};
      headers.forEach((h, i) => {
        obj[h.trim()] = (row[i] || '').trim();
      });
      return obj;
    })
    .filter(n => n['Name'] && n['Name'] !== '')
    .map(n => ({
      tier:        parseInt(n['Tier'] || '0', 10),
      name:        n['Name'],
      deps:        parseDeps(n['Dependencies']),
      rels:        parseRels(n['Relationship']),
      type:        n['Type'] || '',
      description: n['Description'] || '',
      technique:   n['Key Technique'] || '',
      architecture:n['Architecture'] || '',
      objective:   n['(Pre)-training Objective'] || '',
      bibtex:      n['BibTex (Needs Checking)'] || '',
    }));
}

function parseDeps(str) {
  if (!str || str === 'N/A') return [];
  return str.split(';').map(s => s.trim()).filter(Boolean);
}

function parseRels(str) {
  if (!str || str === 'N/A') return [];
  return str.split(';').map(s => s.trim()).filter(Boolean);
}
