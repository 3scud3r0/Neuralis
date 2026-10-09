import { cp, mkdir, readdir, stat, writeFile } from 'node:fs/promises';
import { dirname, extname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const destination = join(root, 'apps/web/dist');
const excluded = new Set(['node_modules', '__pycache__', '.venv', '.pytest_cache', 'runs', 'checkpoints', 'recordings', 'tests', 'target']);
const extensions = new Set(['.html', '.css', '.js', '.mjs', '.ts', '.tsx', '.py', '.rs', '.json', '.toml', '.yaml', '.yml', '.md', '.txt', '.png', '.svg', '.csv']);

async function copyTree(source, target) {
  for (const entry of await readdir(source, { withFileTypes: true })) {
    if (entry.name.startsWith('.') || excluded.has(entry.name)) continue;
    const path = join(source, entry.name);
    const output = join(target, entry.name);
    if (entry.isDirectory()) await copyTree(path, output);
    else if (extensions.has(extname(entry.name))) {
      const size = (await stat(path)).size;
      if (size > 8 * 1024 * 1024) {
        console.warn(`Not publishing oversized demo artifact: ${relative(root, path)}`);
        continue;
      }
      await mkdir(dirname(output), { recursive: true });
      await cp(path, output);
    }
  }
}

await mkdir(destination, { recursive: true });
await copyTree(join(root, 'demos'), join(destination, 'demos'));
await copyTree(join(root, 'screenshots'), join(destination, 'screenshots'));
await writeFile(join(destination, '.nojekyll'), '');
console.log('Packaged runnable demo sources, measured Mario report and screenshot evidence into the static build.');
