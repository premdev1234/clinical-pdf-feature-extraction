// jsons_to_grouped_csv.js
// Produces a grouped CSV where each disease name is on its own line,
// followed by lines "index,url" for that disease.

const fs = require("fs");
const path = require("path");

const jsonDir = path.join(__dirname, "data", "json");
const outFile = path.join(__dirname, "data", "grouped_diseases.csv");

function csvEscape(s) {
  if (s == null) return "";
  const str = String(s);
  if (/[,"\r\n]/.test(str)) return `"${str.replace(/"/g, '""')}"`;
  return str;
}

if (!fs.existsSync(jsonDir)) {
  console.error("JSON directory not found:", jsonDir);
  process.exit(1);
}

const files = fs.readdirSync(jsonDir).filter(f => f.endsWith(".json"));
if (!files.length) {
  console.error("No .json files found in", jsonDir);
  process.exit(1);
}

let linesWritten = 0;
fs.writeFileSync(outFile, "", "utf8");

for (const fn of files) {
  const basename = path.basename(fn, ".json");
  const diseaseName = basename.replace(/_/g, " ");

  fs.appendFileSync(outFile, csvEscape(diseaseName) + "\n", "utf8");
  linesWritten++;

  const fullPath = path.join(jsonDir, fn);
  let content;
  try {
    content = fs.readFileSync(fullPath, "utf8");
  } catch (err) {
    console.warn("Failed to read", fullPath, err.message);
    continue;
  }

  let items;
  try {
    items = JSON.parse(content);
    if (!Array.isArray(items)) {
      if (items && typeof items === "object") {
        if (Array.isArray(items.items)) items = items.items;
        else if (Array.isArray(items.results)) items = items.results;
        else items = [];
      } else {
        items = [];
      }
    }
  } catch (e) {
    console.warn("Skipping invalid JSON (parse error):", fn, e.message);
    continue;
  }

  items.forEach((it, idx) => {
    const link = it.url || it.link || it.href || "";
    const row = `${csvEscape(idx + 1)},${csvEscape(link)}\n`;
    fs.appendFileSync(outFile, row, "utf8");
    linesWritten++;
  });

  fs.appendFileSync(outFile, "\n", "utf8");
  linesWritten++;
}

console.log(`Wrote ${outFile} (${linesWritten} lines, ${files.length} disease files processed).`);
