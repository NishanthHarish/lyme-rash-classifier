"""Score the exported TF.js model by running the real web page in headless Chromium on held-out test images.

usage: browser_eval.py [N_OUT_OF_SCOPE] [N_FLOWERS] [N_RASH]   (run from the working directory; needs `pip install playwright`)
"""
import json, random, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from playwright.sync_api import sync_playwright

W = Path('.'); T = W / 'site_test'
n_oos, n_fl = (int(a) for a in (sys.argv[1:3] if len(sys.argv) > 2 else (250, 100)))
n_rash = int(sys.argv[3]) if len(sys.argv) > 3 else None
CHROME = shutil.which('chromium') or '/opt/pw-browsers/chromium'

shutil.rmtree(T, ignore_errors=True); shutil.copytree(Path(__file__).resolve().parents[2] / 'web-app', T); (T / 'img').mkdir()
split = json.load(open(Path(__file__).with_name('split.json')))
items = json.load(open(W / 'manifest.json'))
lab = {i['path']: i['label'] for i in items}
random.seed(0)
test = ([p for p in split['test'] if lab[p] != 'Other'][:n_rash]
        + random.sample([p for p in split['test'] if lab[p] == 'Other'], n_oos)
        + random.sample([i['path'] for i in items if i.get('source') == 'flowers'], n_fl))
names = []
for k, p in enumerate(test):
    n = f'{k}{Path(p).suffix}'; shutil.copy(p, T / 'img' / n); names.append(n)

JS = '''async names => {
  const res = [];
  for (const n of names) {
    const img = new Image(); img.src = 'img/' + n; await img.decode();
    const S = info.image_size;
    res.push(Array.from(tf.tidy(() => model.predict(
      tf.image.resizeBilinear(tf.browser.fromPixels(img), [S, S], false, true).toFloat().expandDims(0)).dataSync())));
  }
  return res; }'''

srv = subprocess.Popen([sys.executable, '-m', 'http.server', '8766'], cwd=T,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
try:
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=CHROME)
        pg = b.new_page()
        pg.goto('http://localhost:8766/index.html')
        for _ in range(120):
            if pg.evaluate("document.getElementById('statusText').textContent").startswith('Ready'):
                break
            time.sleep(5)
        print('backend', pg.evaluate('tf.getBackend()'), flush=True)
        t0 = time.time(); out = []
        for k in range(0, len(names), 20):
            out += pg.evaluate(JS, names[k:k + 20])
            print(f'{len(out)}/{len(names)} done, {(time.time() - t0) / len(out):.1f} s per image', flush=True)
        b.close()
finally:
    srv.terminate()

P = np.array(out); info = json.load(open(T / 'model-info.json'))
y = np.array([{'Negative': 0, 'Positive': 1, 'Other': 2}[lab[p]] for p in test])
r = y < 2; s = P[r, 1] / (P[r, 0] + P[r, 1]); pred = s >= info['lyme_threshold']; yr = y[r]
print(f"browser rash test n={r.sum()}: accuracy {(pred == yr).mean():.3f}  Lyme recall {pred[yr == 1].mean():.3f}  "
      f"specificity {(~pred[yr == 0]).mean():.3f}  wrongly rejected {(P[r, 2] >= .5).mean():.3f}")
print(f"browser out-of-scope n={(~r).sum()}: rejected {(P[~r, 2] >= .5).mean():.3f}")
print("keras rash test:", json.dumps(info['metrics']['rash_test'])[:300])
np.save(W / 'browser_probs.npy', P)
