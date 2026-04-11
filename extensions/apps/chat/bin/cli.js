#!/usr/bin/env node

import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');
const backendDir = join(rootDir, 'backend');
const frontendDir = join(rootDir, 'frontend');

const args = process.argv.slice(2);
const backendPort = args.includes('--backend-port')
  ? args[args.indexOf('--backend-port') + 1]
  : '8001';
const frontendPort = args.includes('--frontend-port')
  ? args[args.indexOf('--frontend-port') + 1]
  : '3000';

const processes = [];

function cleanup() {
  console.log('\nShutting down...');
  processes.forEach((p) => p.kill());
  process.exit(0);
}

process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

// Start backend
console.log(`Starting backend on http://localhost:${backendPort}`);
const backend = spawn('node', ['src/index.js'], {
  cwd: backendDir,
  env: { ...process.env, PORT: backendPort },
  stdio: 'inherit',
});
processes.push(backend);

// Start frontend
console.log(`Starting frontend on http://localhost:${frontendPort}`);
const frontend = spawn('npm', ['run', 'dev', '--', '-p', frontendPort], {
  cwd: frontendDir,
  stdio: 'inherit',
});
processes.push(frontend);

console.log('');
console.log('Underthesea Chat is running!');
console.log(`  Frontend: http://localhost:${frontendPort}`);
console.log(`  Backend:  http://localhost:${backendPort}`);
console.log('');
console.log('Press Ctrl+C to stop');
