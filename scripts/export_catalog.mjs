// Exports the Open-Generative-AI model catalogs to JSON so the Streamlit app
// can consume them. Usage: node scripts/export_catalog.mjs [repoDir] [outFile]
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');
const repoDir = path.resolve(process.argv[2] || path.join(root, 'Open-Generative-AI'));
const outFile = path.resolve(process.argv[3] || path.join(root, 'data', 'catalog.json'));

const modelsJs = path.join(repoDir, 'packages', 'studio', 'src', 'models.js');
const localCatalogJs = path.join(repoDir, 'electron', 'lib', 'modelCatalog.js');

const m = await import(pathToFileURL(modelsJs).href);
const require = createRequire(import.meta.url);
const { LOCAL_MODEL_CATALOG, ZIMAGE_AUXILIARY } = require(localCatalogJs);

const categories = {
    t2i: m.t2iModels,
    i2i: m.i2iModels,
    t2v: m.t2vModels,
    i2v: m.i2vModels,
    v2v: m.v2vModels,
    lipsync: m.lipsyncModels,
    audio: m.audioModels,
    recast: m.recastModels,
    motion_control: m.motionControlModels,
};

const out = {
    generated_at: new Date().toISOString(),
    repo_dir: repoDir,
    muapi: Object.fromEntries(
        Object.entries(categories).map(([k, v]) => [k, JSON.parse(JSON.stringify(v))])
    ),
    sdcpp: {
        models: LOCAL_MODEL_CATALOG,
        auxiliary: ZIMAGE_AUXILIARY,
    },
};

// A few upstream descriptions contain UTF-8 text that was decoded as CP437.
const MOJIBAKE = { 'ΓÇö': '—', 'ΓÇô': '–', 'ΓÇÖ': '’', 'ΓÇ£': '“', 'ΓÇ¥': '”', 'ΓÇª': '…' };
let json = JSON.stringify(out, null, 1);
for (const [bad, good] of Object.entries(MOJIBAKE)) json = json.split(bad).join(good);

fs.mkdirSync(path.dirname(outFile), { recursive: true });
fs.writeFileSync(outFile, json);
const counts = Object.entries(out.muapi).map(([k, v]) => `${k}=${v.length}`).join(' ');
console.log(`Wrote ${outFile}\nMuAPI: ${counts}\nsd.cpp local models: ${LOCAL_MODEL_CATALOG.length}`);
