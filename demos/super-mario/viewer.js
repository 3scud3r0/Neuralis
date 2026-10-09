'use strict';

// Static, dependency-free report inspection. No fake or interpolated metrics.
const $ = (id) => document.getElementById(id);
let report = null;
let bundled = true;
let frameIndex = 0;
const number = new Intl.NumberFormat('en-US');
const commands = {
  smoke: 'python doctor.py\npython train.py --config configs/smoke.json\npython -m pytest -q',
  train: 'python train.py --config configs/level-1-1.json\n# Resume without restarting the model:\npython train.py --resume runs/level-1-1/latest.pt --steps 20000000',
  evaluate: 'python evaluate.py runs/level-1-1/latest.pt \\\n  --episodes 30 --seed 100042 --video \\\n  --output runs/level-1-1/evaluation',
  export: 'python export.py runs/level-1-1 \\\n  --evaluation runs/level-1-1/evaluation/evaluation.json \\\n  --inspection runs/level-1-1/evaluation/inspection.json \\\n  --output runs/mario-export.json',
};

function notice(message, error = false) {
  $('notice').textContent = message;
  $('notice').classList.toggle('error', error);
}

function finite(value, fallback = 0) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function format(value, digits = 3) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = String(content);
  return node;
}

function detail(label, value) {
  const node = element('div', 'detail');
  node.append(element('span', '', label), element('b', '', value));
  return node;
}

function validate(data) {
  if (!data || typeof data !== 'object' || data.schema !== 'neuralis.mario.run/v1') {
    throw new Error('Choose a Neuralis Mario report exported by export.py. Expected schema neuralis.mario.run/v1.');
  }
  if (!data.run || !Array.isArray(data.metrics) || data.metrics.length > 100000) {
    throw new Error('Report is missing run metadata or exceeds the 100,000-update browser budget.');
  }
  if (!data.metrics.every((point) => point && typeof point === 'object' && !Array.isArray(point) && Number.isFinite(point.step) && point.step >= 0)) {
    throw new Error('Every optimization sample must be an object with a finite, nonnegative transition count.');
  }
  if (!data.architecture || !Array.isArray(data.architecture.layers) || data.architecture.layers.length > 128) {
    throw new Error('Report is missing its model architecture or exceeds the 128-layer viewer budget.');
  }
  return data;
}

function render(data, isBundled) {
  report = validate(data);
  bundled = isBundled;
  frameIndex = 0;
  $('transitions').textContent = number.format(finite(report.run.totalTransitions));
  $('updates').textContent = `${number.format(report.metrics.length)} measured updates · ${report.run.status || 'unknown status'}`;
  $('parameters').textContent = number.format(finite(report.architecture.parameters));
  const evaluation = report.evaluation;
  if (evaluation) {
    $('success').textContent = `${format(100 * finite(evaluation.successRate), 1)}%`;
    const interval = evaluation.successRate95CI;
    $('confidence').textContent = Array.isArray(interval) ? `95% CI ${format(interval[0] * 100, 1)}–${format(interval[1] * 100, 1)}%` : 'Confidence interval unavailable';
    $('max-x').textContent = number.format(finite(evaluation.maxX));
    $('evaluation-count').textContent = `${evaluation.episodes} evaluated episodes · seed ${evaluation.seed}`;
    $('evaluation-note').textContent = evaluation.note || 'Success is measured from flag_get, not predicted from reward.';
  } else {
    $('success').textContent = '—';
    $('max-x').textContent = '—';
    $('confidence').textContent = 'No evaluation loaded';
    $('evaluation-count').textContent = 'Export an evaluation to attach performance';
  }
  const slider = $('update-index');
  slider.max = String(Math.max(0, report.metrics.length - 1));
  slider.value = slider.max;
  slider.disabled = report.metrics.length === 0;
  renderUpdate();
  renderArchitecture();
  renderEpisodes();
  renderLayers();
  renderFrame();
  drawChart();
  notice(isBundled
    ? 'Captured smoke test: real NES rollouts and PPO optimization, followed by bounded evaluation. This policy has not solved Mario. Train longer on your hardware to measure progress.'
    : `Imported ${report.run.id || 'local run'}. All metrics are read directly from your exported experiment; frame images remain in your local evaluation directory.`);
}

function renderUpdate() {
  const index = Number($('update-index').value);
  const metric = report?.metrics[index];
  const container = $('update-details');
  container.replaceChildren();
  if (!metric) return;
  container.append(
    detail('Selected transition', number.format(finite(metric.step))),
    detail('Approximate KL', format(metric.approxKL, 5)),
    detail('Action entropy', format(metric.entropy)),
    detail('Learning rate', format(metric.learningRate, 6)),
  );
}

function renderArchitecture() {
  const container = $('architecture');
  container.replaceChildren();
  report.architecture.layers.forEach((layer) => {
    const node = element('div', 'layer-node');
    node.append(element('span', 'layer-icon'));
    node.append(element('strong', '', layer.type || 'Layer'));
    node.append(element('small', '', layer.name || ''));
    node.append(element('small', '', Array.isArray(layer.outputShape) ? layer.outputShape.join(' × ') : ''));
    node.append(element('small', '', `${number.format(finite(layer.parameters))} params`));
    container.append(node);
  });
}

function renderEpisodes() {
  const body = $('episodes');
  body.replaceChildren();
  const episodes = report.evaluation?.results || [];
  episodes.slice(0, 1000).forEach((episode, index) => {
    const row = element('tr');
    const outcome = episode.flagGet ? 'Flag reached' : episode.truncated ? 'Time limit' : 'Death / terminal';
    row.append(
      element('td', '', String(episode.episode ?? index + 1).padStart(2, '0')),
      element('td', '', format(episode.reward, 1)),
      element('td', '', number.format(finite(episode.xPos))),
      element('td', '', number.format(finite(episode.length))),
      element('td', episode.flagGet ? 'success' : 'timeout', outcome),
    );
    body.append(row);
  });
  if (episodes.length === 0) {
    const row = element('tr');
    const cell = element('td', '', 'No measured evaluation in this report.');
    cell.colSpan = 5;
    row.append(cell);
    body.append(row);
  }
}

function renderLayers() {
  const select = $('layer');
  select.replaceChildren();
  const layers = report.inspection?.layers || [];
  layers.forEach((layer, index) => {
    const option = element('option', '', layer.layer);
    option.value = String(index);
    select.append(option);
  });
  select.disabled = layers.length === 0;
  renderActivation();
}

function renderActivation() {
  const layer = report?.inspection?.layers?.[Number($('layer').value)];
  const grid = $('activation-grid');
  const details = $('activation-details');
  grid.replaceChildren();
  details.replaceChildren();
  if (!layer) {
    grid.append(element('p', 'muted', 'Attach inspection.json with export.py to inspect actual layer values.'));
    return;
  }
  const values = Array.isArray(layer.sample) ? layer.sample.slice(0, 128).map((value) => finite(value)) : [];
  const max = Math.max(1e-8, ...values.map(Math.abs));
  values.forEach((value, index) => {
    const cell = element('div', 'activation-cell');
    const normalized = value / max;
    cell.style.background = normalized >= 0
      ? `hsl(185 ${45 + 50 * normalized}% ${10 + 57 * normalized}%)`
      : `hsl(275 ${45 + 45 * -normalized}% ${10 + 55 * -normalized}%)`;
    cell.title = `Activation ${index}: ${value.toFixed(6)}`;
    grid.append(cell);
  });
  details.append(detail('Mean', format(layer.mean, 5)), detail('Std deviation', format(layer.std, 5)),
                 detail('Minimum', format(layer.min, 5)), detail('Maximum', format(layer.max, 5)));
}

function renderFrame() {
  const captures = report?.evaluation?.captures || [];
  const image = $('frame');
  if (!bundled || captures.length === 0) {
    image.hidden = true;
    image.removeAttribute('src');
    $('frame-label').textContent = bundled ? 'No captured frames available' : 'Open numbered PNGs from your evaluation output directory';
    $('frame-position').textContent = 'Local artifacts';
    $('previous-frame').disabled = true;
    $('next-frame').disabled = true;
    return;
  }
  frameIndex = (frameIndex + captures.length) % captures.length;
  const capture = captures[frameIndex];
  // Restrict bundled paths to local PNG filenames; imported reports cannot load URLs.
  if (!/^[a-zA-Z0-9_-]+\.png$/.test(capture.file)) return;
  image.hidden = false;
  image.src = `evidence/${capture.file}`;
  $('frame-label').textContent = `Episode ${capture.episode} · action step ${capture.step} · X ${capture.xPos}`;
  $('frame-position').textContent = `${String(frameIndex + 1).padStart(2, '0')} / ${String(captures.length).padStart(2, '0')}`;
  $('previous-frame').disabled = false;
  $('next-frame').disabled = false;
}

function drawChart() {
  if (!report) return;
  const canvas = $('chart');
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (width < 1 || height < 1) return;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  const padding = {left: 52, right: 14, top: 18, bottom: 24};
  const metric = $('metric').value;
  const points = report.metrics.filter((point) => typeof point[metric] === 'number' && Number.isFinite(point[metric]));
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  if (points.length === 0) {
    ctx.fillStyle = '#889ab5';
    ctx.font = '12px system-ui';
    ctx.fillText('No metric samples are available.', padding.left, height / 2);
    return;
  }
  const values = points.map((point) => point[metric]);
  let low = Math.min(...values);
  let high = Math.max(...values);
  if (high === low) { high += 0.5; low -= 0.5; }
  const buffer = (high - low) * 0.1;
  low -= buffer;
  high += buffer;
  const first = finite(points[0].step);
  const last = finite(points[points.length - 1].step);
  const x = (point) => padding.left + (finite(point.step) - first) / Math.max(1, last - first) * chartWidth;
  const y = (point) => padding.top + chartHeight - (point[metric] - low) / (high - low) * chartHeight;
  ctx.font = '9px system-ui';
  ctx.textAlign = 'right';
  for (let tick = 0; tick <= 4; tick++) {
    const coordinate = padding.top + chartHeight * tick / 4;
    ctx.strokeStyle = '#1c2b43';
    ctx.beginPath(); ctx.moveTo(padding.left, coordinate); ctx.lineTo(width - padding.right, coordinate); ctx.stroke();
    ctx.fillStyle = '#7a8ca8';
    ctx.fillText((high - (high - low) * tick / 4).toPrecision(3), padding.left - 9, coordinate + 3);
  }
  ctx.beginPath();
  points.forEach((point, index) => index === 0 ? ctx.moveTo(x(point), y(point)) : ctx.lineTo(x(point), y(point)));
  ctx.lineTo(x(points[points.length - 1]), padding.top + chartHeight);
  ctx.lineTo(x(points[0]), padding.top + chartHeight);
  ctx.closePath();
  const gradient = ctx.createLinearGradient(0, padding.top, 0, padding.top + chartHeight);
  gradient.addColorStop(0, '#45dce63a'); gradient.addColorStop(1, '#45dce602');
  ctx.fillStyle = gradient; ctx.fill();
  ctx.beginPath();
  points.forEach((point, index) => index === 0 ? ctx.moveTo(x(point), y(point)) : ctx.lineTo(x(point), y(point)));
  ctx.strokeStyle = '#45dce6'; ctx.lineWidth = 2; ctx.stroke();
  const selected = report.metrics[Number($('update-index').value)];
  if (selected && Number.isFinite(selected[metric])) {
    ctx.fillStyle = '#dcfcff';
    ctx.beginPath(); ctx.arc(x(selected), y(selected), 4, 0, Math.PI * 2); ctx.fill();
  }
  $('chart-range').textContent = `${number.format(first)} → ${number.format(last)} transitions · ${points.length} actual samples`;
}

$('metric').addEventListener('change', drawChart);
$('layer').addEventListener('change', renderActivation);
$('update-index').addEventListener('input', () => { renderUpdate(); drawChart(); });
$('previous-frame').addEventListener('click', () => { frameIndex--; renderFrame(); });
$('next-frame').addEventListener('click', () => { frameIndex++; renderFrame(); });
$('report-file').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  if (file.size > 25 * 1024 * 1024) { notice('Report exceeds the 25 MiB browser import limit.', true); return; }
  try { render(JSON.parse(await file.text()), false); }
  catch (error) { notice(error.message, true); }
});
document.querySelectorAll('[data-command]').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-command]').forEach((node) => node.classList.remove('selected'));
    button.classList.add('selected');
    $('command').textContent = commands[button.dataset.command];
  });
});
$('copy-command').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText($('command').textContent);
    $('copy-command').textContent = 'Copied';
    window.setTimeout(() => { $('copy-command').textContent = 'Copy command'; }, 1500);
  } catch { notice('Clipboard is unavailable. Select and copy the command text manually.', true); }
});
window.addEventListener('resize', drawChart);
$('command').textContent = commands.smoke;
fetch('evidence/sample-run.json').then((response) => {
  if (!response.ok) throw new Error(`Sample report request failed (${response.status}). Serve this directory over HTTP.`);
  return response.json();
}).then((data) => render(data, true)).catch((error) => notice(error.message, true));
