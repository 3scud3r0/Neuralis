import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = join(root, 'screenshots');
const base = process.env.NEURALIS_CAPTURE_URL || 'http://127.0.0.1:4173';
const browser = await chromium.launch({
  executablePath: process.env.NEURALIS_CHROMIUM || '/usr/bin/chromium',
  headless: true,
  args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
});
const context = await browser.newContext({ viewport: { width: 1600, height: 1050 }, deviceScaleFactor: 1 });
const page = await context.newPage();
const errors = [];
page.on('pageerror', error => errors.push(error.message));
await mkdir(output, { recursive: true });
const manifest = [];
const screens = [
  ['overview', '/', 'Research overview'],
  ['demo-library', '/demos', 'Runnable browser demo library'],
  ['training', '/training', 'Real browser model training'],
  ['network-explorer', '/visualization', 'Interactive neural network explorer'],
  ['tensor-inspector', '/tensors', 'Measured weights, gradients and activations'],
  ['attention', '/attention', 'Attention inspection and capability states'],
  ['interventions', '/interventions', 'Controlled activation interventions'],
  ['experiments', '/experiments', 'Recorded experiments and replay'],
  ['comparison', '/compare', 'Run comparison'],
  ['datasets', '/data', 'Dataset exploration and custom CSV import'],
  ['infrastructure', '/lab', 'Actual Python runtime connection'],
  ['settings', '/settings', 'Workspace and desktop runtime settings'],
  ['new-experiment', '/new-experiment', 'Experiment configuration wizard'],
  ['artifacts', '/artifacts', 'Checkpoint and artifact management'],
  ['readiness', '/readiness', 'Execution and capability diagnostics'],
  ['representations', '/representations', 'Representation analysis workbench'],
  ['dataset-preparation', '/datasets', 'Leakage-safe CSV dataset preparation'],
];
try {
  await page.goto(`${base}/#/training`, { waitUntil: 'networkidle' });
  await page.getByRole('button', { name: 'Start training', exact: true }).first().click();
  await page.waitForTimeout(2000);
  const pause = page.getByRole('button', { name: 'Pause training', exact: true }).first();
  if (await pause.count()) await pause.click();
  const checkpoint = page.getByRole('button', { name: 'Checkpoint', exact: true });
  if (await checkpoint.count()) await checkpoint.first().click();
  for (let index = 0; index < screens.length; index++) {
    const [slug, route, title] = screens[index];
    await page.evaluate(next => { window.location.hash = next; }, route);
    await page.waitForTimeout(route === '/visualization' ? 1700 : 500);
    const filename = `${String(index + 1).padStart(3, '0')}-${slug}.png`;
    await page.screenshot({ path: join(output, filename), fullPage: true, animations: 'disabled' });
    manifest.push({ number: index + 1, file: filename, route, title, capturedAt: new Date().toISOString(), kind: 'actual-application-screenshot' });
    console.log(filename);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${base}/#/demos`, { waitUntil: 'networkidle' });
  await page.screenshot({ path: join(output, '018-mobile-demo-library.png'), fullPage: true, animations: 'disabled' });
  manifest.push({ number: 18, file: '018-mobile-demo-library.png', route: '/demos', title: 'Responsive mobile demo library', capturedAt: new Date().toISOString(), kind: 'actual-application-screenshot' });
  if (errors.length) throw new Error(`Browser errors during capture: ${[...new Set(errors)].join('; ')}`);
  await writeFile(join(output, 'manifest.json'), JSON.stringify({ viewport: { width: 1600, height: 1050 }, source: 'local built application', screenshots: manifest, pageErrors: errors }, null, 2) + '\n');
  const cards = manifest.map(item => `<figure><a href="${item.file}"><img src="${item.file}" alt="${item.title}" loading="lazy"></a><figcaption>${String(item.number).padStart(3, '0')} — ${item.title}</figcaption></figure>`).join('\n');
  await writeFile(join(output, 'index.html'), `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Neuralis Studio — verified screenshots</title><style>body{margin:0;padding:40px;background:#070f1d;color:#d7e5f6;font:15px system-ui}h1{font-size:32px}p{color:#91a6c0}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:24px}figure{margin:0;background:#101d30;border:1px solid #2b405b;border-radius:12px;overflow:hidden}img{width:100%;display:block}figcaption{padding:15px}a{color:#71ddec}</style><h1>Neuralis Studio</h1><p>Numbered captures of the implemented application. These are actual browser screenshots, not generated mockups. Measured Mario evidence is in the separate demo folder.</p><p><a href="../index.html">Open the research workbench</a></p><main>${cards}</main></html>`);
} finally {
  await context.close();
  await browser.close();
}
