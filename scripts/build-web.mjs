// Builds the static site in web/: the WebAssembly engine (rust/cloud-map-generator-wasm) and the page of ui/static.
//
//   node scripts/build-web.mjs            build everything (release wasm)
//   node scripts/build-web.mjs --page     only the page files (index.html, app.js); no cargo, no wasm-bindgen
//
// Needs: rustc with the wasm32-unknown-unknown target, and the `wasm-bindgen` CLI whose version equals the
// wasm-bindgen crate in rust/Cargo.lock (cargo install wasm-bindgen-cli --version <that version>). `wasm-opt`
// (binaryen) is used when it is on the PATH. The Rust build goes to rust/target/web, apart from rust/target.
//
// web/ keeps its hand-written files (engine.js, worker.js, README.md) and receives what this script generates,
// which is not committed (web/.gitignore): index.html and app.js are copies of ui/static's, so the page that
// `cloud-map-generator serve` shows and the page the site shows cannot drift apart; pkg/ is the engine.

import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const rust = join(root, 'rust');
const web = join(root, 'web');
const run = (cmd, args, opts = {}) => execFileSync(cmd, args, { stdio: 'inherit', ...opts });
const out = (cmd, args) => execFileSync(cmd, args, { encoding: 'utf8' }).trim();

function buildPage() {
  copyFileSync(join(root, 'ui/static/app.js'), join(web, 'app.js'));
  let html = readFileSync(join(root, 'ui/static/index.html'), 'utf8');
  // Relative paths, so the site works under https://<user>.github.io/<repo>/ as well as at a root.
  const module = '<script type="module" src="/static/app.js"></script>';
  if (!html.includes(module)) throw new Error('ui/static/index.html no longer loads /static/app.js: update scripts/build-web.mjs');
  html = html.replace(module, '<script type="module" src="./app.js"></script>');
  // The engine adapter must be in place before the page's first call to /api/defaults.
  const head = '</head>';
  if (!html.includes(head)) throw new Error('ui/static/index.html has no </head>: update scripts/build-web.mjs');
  html = html.replace(head, `  <script src="./engine.js"></script>\n${head}`);
  writeFileSync(join(web, 'index.html'), html);
  console.log('page: web/index.html, web/app.js');
}

function crateVersion(name) {
  const lock = readFileSync(join(rust, 'Cargo.lock'), 'utf8');
  const m = new RegExp(`name = "${name}"\r?\nversion = "([^"]+)"`).exec(lock);
  if (!m) throw new Error(`${name} is not in rust/Cargo.lock`);
  return m[1];
}

function buildWasm() {
  const want = crateVersion('wasm-bindgen');
  const have = out('wasm-bindgen', ['--version']).split(' ')[1];
  if (want !== have) {
    throw new Error(`wasm-bindgen CLI ${have} does not match the crate ${want} in rust/Cargo.lock: cargo install wasm-bindgen-cli --version ${want} --locked`);
  }
  run('cargo', ['build', '-p', 'cloud-map-generator-wasm', '--target', 'wasm32-unknown-unknown', '--release', '--target-dir', 'target/web'], { cwd: rust });
  const wasm = join(rust, 'target/web/wasm32-unknown-unknown/release/cloud_map_generator_wasm.wasm');
  const pkg = join(web, 'pkg');
  rmSync(pkg, { recursive: true, force: true });
  mkdirSync(pkg, { recursive: true });
  run('wasm-bindgen', ['--target', 'web', '--no-typescript', '--out-dir', pkg, '--out-name', 'cloud_map_generator_wasm', wasm]);
  const bg = join(pkg, 'cloud_map_generator_wasm_bg.wasm');
  try {
    run('wasm-opt', ['-O3', '--enable-bulk-memory', '--enable-nontrapping-float-to-int', '--enable-sign-ext', '--enable-mutable-globals', '-o', bg, bg]);
    console.log('wasm-opt: applied');
  } catch {
    console.log('wasm-opt: not available, the module is shipped as the linker made it');
  }
  console.log(`wasm: web/pkg/cloud_map_generator_wasm_bg.wasm, ${(statSync(bg).size / 1e6).toFixed(2)} MB`);
}

if (!existsSync(web)) mkdirSync(web, { recursive: true });
if (!process.argv.includes('--page')) buildWasm();
buildPage();
