import { chromium } from 'playwright';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { spawn } from 'node:child_process';

const root = resolve(import.meta.dirname, '..');
const directory = resolve(root, 'screenshots');
const base = process.env.NEURALIS_CAPTURE_URL || 'http://127.0.0.1:4173';
const backend = process.env.NEURALIS_CAPTURE_BACKEND || 'http://127.0.0.1:8001';
const browser = await chromium.launch({executablePath: process.env.NEURALIS_CHROMIUM || '/usr/bin/chromium',headless:true,
  args:['--no-sandbox','--enable-unsafe-webgpu','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page = await browser.newPage({viewport:{width:1600,height:1050},deviceScaleFactor:1});
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const captures = [];
let recorder = null;
const route = async path => {await page.evaluate(next => {window.location.hash = next;},path);await page.waitForTimeout(400);};
async function capture(number, slug, title, extra={}) {
  await page.evaluate(() => window.scrollTo(0,0));
  await page.waitForTimeout(500);
  const file = `${String(number).padStart(3,'0')}-${slug}.png`;
  await page.screenshot({path:resolve(directory,file),fullPage:true,animations:'disabled'});
  captures.push({number,file,title,route:new URL(page.url()).hash.slice(1),capturedAt:new Date().toISOString(),kind:'actual-application-screenshot',...extra});
  console.log(file);
}
try {
  await page.goto(`${base}/#/research`,{waitUntil:'networkidle'});
  await page.getByRole('button',{name:'Connect backend',exact:true}).click();
  await page.getByLabel('Extension backend URL').fill(backend);
  await page.getByRole('button',{name:'Check capabilities',exact:true}).click();
  await page.getByText('AXISLAB · available',{exact:true}).waitFor();
  const axisResponse = page.waitForResponse(response => response.url().endsWith('/extensions/axis/evaluate'));
  await page.getByRole('button',{name:'Evaluate with AxisLab backend',exact:true}).click();
  if ((await axisResponse).status() !== 200) throw new Error('AxisLab evaluation failed.');
  await page.waitForTimeout(300);
  await capture(19,'axislab','AxisLab paired directions and held-out evaluation');
  await page.getByRole('tab',{name:/Efficient training/}).click();
  await capture(20,'nf4','Packed NF4 quantization and reconstruction');
  await page.getByRole('button',{name:'Run CPU adaptation',exact:true}).click();
  await page.getByText('Backend method:',{exact:false}).waitFor();
  await capture(21,'lora','Measured frozen-base low-rank adaptation');
  await page.getByRole('tab',{name:/Data Foundry/}).click();
  await page.getByRole('button',{name:'Generate and validate',exact:true}).click();
  await capture(22,'data-foundry','Verified synthetic JSONL generation and split');
  await page.getByRole('tab',{name:/Local RAG/}).click();
  await page.getByLabel('RAG document text').fill('Low rank adapters preserve frozen base weights while learning small trainable matrices. Neuralis stores encrypted activation research documents locally.');
  await page.getByRole('button',{name:'Send editor text to backend',exact:true}).click();
  await page.getByText('Backend indexing response',{exact:true}).waitFor();
  await page.getByRole('button',{name:'Search backend collection',exact:true}).click();
  await page.getByText('Backend search response',{exact:true}).waitFor();
  await capture(23,'local-rag','Local retrieval with actual encrypted backend collection');
  await page.getByRole('tab',{name:/RL Arena/}).click();
  await page.getByRole('button',{name:'Train and record rollout',exact:true}).click();
  await page.getByRole('button',{name:'Train and record rollout',exact:true}).waitFor();
  await capture(24,'rl-arena','Actual tabular policy training and trajectory replay');
  await page.getByRole('button',{name:'Load recorded episodes',exact:true}).click();
  await page.getByLabel('Recorded episode').selectOption('mario-cli-ppo-confirmed-20261009');
  await page.locator('.arena-recorded-image').waitFor();
  await capture(25,'mario-temporal-debugger','Actual NES pixels and persisted policy evidence');
  // A live recorder is optional: run it only when the validated local checkpoint exists.
  if (process.env.NEURALIS_CAPTURE_MARIO === '1') {
    recorder = spawn('/workspace/.venvs/mario/bin/python',['arena.py','runs/smoke/latest.pt','--url',backend,'--episode-id','mario-browser-live-'+Date.now(),'--max-steps','50','--fps','2'],{cwd:resolve(root,'demos/super-mario'),stdio:['ignore','pipe','pipe']});
    let log='';recorder.stdout.on('data',part => {log+=part.toString();});recorder.stderr.on('data',part => {log+=part.toString();});
    await page.waitForTimeout(3000);
    await page.getByRole('button',{name:'Refresh episodes',exact:true}).click();
    const liveEpisode = await page.getByLabel('Live episode').locator('option').evaluateAll(options => options.map(option=>option.value).find(value=>value.startsWith('mario-browser-live-')));
    if (!liveEpisode) throw new Error('The live Mario recorder produced no stored episode.');
    await page.getByLabel('Live episode').selectOption(liveEpisode);
    await page.getByRole('button',{name:'Connect video',exact:true}).click();
    await Promise.race([
      page.waitForFunction(() => {const video=document.querySelector('.arena-live-video');return video && video.readyState>=2 && video.videoWidth>0;},{},{timeout:25000}),
      page.locator('.arena-runtime-card').first().getByRole('alert').waitFor({timeout:25000}),
    ]);
    const received = await page.locator('.arena-live-video').evaluate(video=>video.videoWidth>0);
    if (!received && !(await page.locator('.arena-runtime-card').first().getByRole('alert').innerText()).includes('ICE')) {
      throw new Error('Unexpected video failure instead of a reported ICE restriction.');
    }
    await capture(26,'mario-live-video',received ? 'Actual Mario frames received through browser WebRTC' : 'Actual WebRTC capability diagnosis under managed browser UDP restrictions',
      {transport:received ? 'webrtc' : 'blocked-by-managed-browser-policy',frameSource:'local NES policy recorder',decodedBrowserVideo:received});
    const stop = page.getByRole('button',{name:'Stop',exact:true});
    if(await stop.isEnabled()) await stop.click();
    recorder.kill('SIGINT');recorder=null;
  }
  await route('/demos');
  await page.getByLabel('Search demos').fill('Causal Transformer microscope');
  await page.getByRole('button',{name:'Launch',exact:true}).click();
  await page.getByRole('button',{name:'Start training',exact:true}).first().click();
  await page.waitForTimeout(3000);
  const pause=page.getByRole('button',{name:'Pause training',exact:true}).first();if(await pause.count()) await pause.click();
  await route('/inference');
  await page.getByRole('button',{name:'Generate',exact:true}).click();
  await page.getByText('Token logit lens',{exact:true}).waitFor();
  await page.waitForTimeout(1200);
  await capture(27,'token-xray','Actual trained Transformer inference and residual logit lens');
  await route('/advanced');
  await capture(28,'webgpu-observatory','Actual WGSL activation particle rendering');
  await page.getByRole('button',{name:'Attention splines',exact:true}).click();
  await capture(29,'attention-splines','Weighted curves from captured causal attention');
  await page.getByRole('button',{name:'Latent galaxy',exact:true}).click();
  await page.getByLabel('Representation source').selectOption('embedding');
  await capture(30,'latent-galaxy','Measured PCA of learned vocabulary embeddings');
  await route('/graph');
  await page.getByLabel('Graph backend URL').fill(backend);
  await page.getByRole('button',{name:'Compile and evaluate',exact:true}).click();
  await page.getByText('BACKEND COMPILED',{exact:true}).waitFor();
  await capture(31,'graph-surgery','Actual backend residual DAG compilation and evaluation');
  await page.setViewportSize({width:390,height:844});
  await route('/research');
  await capture(32,'mobile-research-workbench','Responsive research extensions on mobile');
  if(errors.length) throw new Error('Application errors: '+[...new Set(errors)].join('; '));
  const manifest=JSON.parse(await readFile(resolve(directory,'manifest.json'),'utf8'));
  manifest.screenshots=manifest.screenshots.filter(item=>item.number<19).concat(captures);
  manifest.pageErrors=[];
  manifest.extensionValidation={backend:'local PyTorch API',frameEvidence:'actual stored NES captures',browserWebRTC:captures.find(item=>item.number===26)?.transport??'not-attempted',browserErrors:errors};
  await writeFile(resolve(directory,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
  const cards=manifest.screenshots.map(item=>`<figure><a href="${item.file}"><img src="${item.file}" alt="${item.title}" loading="lazy"></a><figcaption>${String(item.number).padStart(3,'0')} — ${item.title}</figcaption></figure>`).join('\n');
  await writeFile(resolve(directory,'index.html'),`<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Neuralis verified screenshots</title><style>body{margin:0;padding:40px;background:#070f1d;color:#d7e5f6;font:15px system-ui}h1{font-size:32px}p{color:#91a6c0}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:24px}figure{margin:0;background:#101d30;border:1px solid #2b405b;border-radius:12px;overflow:hidden}img{width:100%;display:block}figcaption{padding:15px}a{color:#71ddec}</style><h1>Neuralis Studio</h1><p>Numbered actual browser captures. Training, AxisLab, encrypted retrieval, graph evaluation and stored NES pixels come from executed local computations. Live transport restrictions are reported explicitly.</p><p><a href="../index.html">Open the workbench</a></p><main>${cards}</main></html>`);
} finally {recorder?.kill('SIGINT');await browser.close();}
