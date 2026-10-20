// FunnyBirds render server, a faster drop-in for the authors' render/server.js (same /page and
// /render API, see funnybirds_render.py). It serves the authors' page.html and models unchanged,
// except for two text patches checked at startup: a hook that counts GLTF loads in flight, and the
// removal of an import-map polyfill fetched from unpkg.com (Chrome 103 has import maps natively).
// Its images are pixel-identical to the authors' server. (Skipping the frames drawn before the
// models are loaded saves ~10% but changes a few antialiased edge pixels, so the loop is unchanged.)
//
// The authors' server starts a new Chrome per screenshot and waits a fixed second. This one keeps
// one Chrome with a pool of tabs and takes the screenshot when the scene is drawn: after every
// model has loaded (or 404ed, for a dropped part) and two more animation frames. Requests run in
// parallel, up to one per tab.
//
//   node funnybirds_server.js <funnybirds>/render [--port 8081] [--pages N] [--timeout ms]
//
// <funnybirds>/render is the authors' render folder after `npm install`; express, ejs and
// puppeteer are loaded from its node_modules.
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');

const args = process.argv.slice(2);
const opt = (name, dflt) => (args.includes(name) ? args[args.indexOf(name) + 1] : dflt);
const renderDir = path.resolve(args[0] || '.');
const port = Number(opt('--port', 8081));
const nPages = Number(opt('--pages', os.cpus().length));
const timeout = Number(opt('--timeout', 60000));
const need = (m) => require(path.join(renderDir, 'node_modules', m));
const express = need('express');
const ejs = need('ejs');
const puppeteer = need('puppeteer');

// The template variables of page.html, as the authors' server.js passes them.
const KEYS = ['render_mode', 'camera_distance', 'camera_pitch', 'camera_roll', 'light_distance', 'light_pitch',
  'light_roll', 'beak_model', 'beak_color', 'foot_model', 'eye_model', 'tail_model', 'tail_color', 'wing_model',
  'wing_color', 'bg_objects', 'bg_scale_x', 'bg_scale_y', 'bg_scale_z', 'bg_rot_x', 'bg_rot_y', 'bg_rot_z',
  'bg_color', 'bg_radius', 'bg_pitch', 'bg_roll'];
const SHIMS = '<script async src="https://unpkg.com/es-module-shims@1.3.6/dist/es-module-shims.js"></script>';
const IMPORT = "import { TextGeometry } from './jsm/geometries/TextGeometry.js';";
const HOOK = `
			// funnybirds_server.js: count GLTF loads in flight; the scene is complete when none are left.
			window.__fb = { started: false, pending: 0 };
			{
				const load = GLTFLoader.prototype.load;
				GLTFLoader.prototype.load = function ( url, onLoad, onProgress, onError ) {
					window.__fb.started = true;
					window.__fb.pending ++;
					const done = () => { window.__fb.pending --; };
					return load.call( this, url, ( gltf ) => { try { onLoad( gltf ); } finally { done(); } }, onProgress,
						( e ) => { done(); if ( onError ) onError( e ); } );
				};
			}
`;

function patchedTemplate() {
  let html = fs.readFileSync(path.join(renderDir, 'page.html'), 'utf8');
  const at = html.lastIndexOf(IMPORT);
  if (at < 0 || !html.includes(SHIMS)) {
    throw new Error(`${renderDir}/page.html changed: cannot apply the patches (check the funnybirds commit)`);
  }
  html = html.slice(0, at + IMPORT.length) + HOOK + html.slice(at + IMPORT.length);
  return html.replace(SHIMS, '');
}

const template = patchedTemplate();
const app = express();
app.get('/page', (req, res) => {
  res.type('html').send(ejs.render(template, Object.fromEntries(KEYS.map((k) => [k, req.query[k]]))));
});
app.get('/render', async (req, res) => {
  try {
    res.end(await shoot(req.url.split('?')[1] || ''));
  } catch (e) {
    console.error(e);
    res.status(500).end(String(e));
  }
});
app.use(express.static(path.join(renderDir, 'js', 'three.js-master', 'examples')));

let browser;
const free = [];
const waiting = [];

async function newPage() {
  const page = await browser.newPage();
  await page.setViewport({ width: 1256, height: 1256 }); // as the authors' server; images are resized later
  return page;
}

function acquire() {
  return free.length ? Promise.resolve(free.pop()) : new Promise((resolve) => waiting.push(resolve));
}

function release(page) {
  const next = waiting.shift();
  if (next) next(page);
  else free.push(page);
}

// Base64 PNG of one scene. Idle tabs go to about:blank so their render loops stop using the CPU.
async function shoot(query) {
  let page = await acquire();
  try {
    await page.goto(`http://localhost:${port}/page?${query}`, { waitUntil: 'load', timeout });
    await page.waitForFunction('window.__fb && window.__fb.started && window.__fb.pending === 0',
      { polling: 'raf', timeout });
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    const png = await page.screenshot({ encoding: 'base64' });
    await page.goto('about:blank');
    return png;
  } catch (e) {
    await page.close().catch(() => {});
    page = await newPage(); // a timed-out or crashed tab is replaced, not reused
    throw e;
  } finally {
    release(page);
  }
}

(async () => {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'funnybirds-chrome-'));
  browser = await puppeteer.launch({
    headless: true,
    userDataDir: profile,
    args: ['--use-gl=swiftshader', '--no-sandbox', '--enable-surface-synchronization'],
  });
  for (let i = 0; i < nPages; i++) free.push(await newPage());
  const server = app.listen(port, () => console.log(`FunnyBirds render server: http://localhost:${port}, ${nPages} pages`));
  const stop = async () => {
    server.close();
    await browser.close().catch(() => {});
    fs.rmSync(profile, { recursive: true, force: true });
    process.exit(0);
  };
  process.on('SIGTERM', stop);
  process.on('SIGINT', stop);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
