// Browser end-to-end tests: real React app + real FastAPI backend + real search models,
// on a throwaway stack (see backend/scripts/e2e_server.py). Run with:  npm run e2e
import path from 'node:path'
import { defineConfig, devices } from '@playwright/test'

const BACKEND_PORT = 8011
const FRONTEND_PORT = 5181
// Absolute path with forward slashes: Playwright's command parser treats backslashes as escapes.
const venv = path.resolve(import.meta.dirname, '..', '.venv').split(path.sep).join('/')
const python =
  process.env.E2E_PYTHON || (process.platform === 'win32' ? `${venv}/Scripts/python.exe` : `${venv}/bin/python`)

export default defineConfig({
  testDir: './e2e',
  timeout: 120_000, // the first test loads the embedding + reranker models
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1, // one shared backend; tests use their own accounts
  retries: 0,
  reporter: [['list'], ['html', { open: 'never', outputFolder: 'e2e-report' }]],
  outputDir: 'e2e-results',
  use: {
    baseURL: `http://localhost:${FRONTEND_PORT}`,
    // Locally: the Microsoft Edge already installed on Windows (no browser download).
    // In CI (Linux): Playwright's Chromium, installed by the workflow.
    ...(process.env.CI ? devices['Desktop Chrome'] : { ...devices['Desktop Edge'], channel: 'msedge' }),
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: [
    {
      command: `"${python}" scripts/e2e_server.py --port ${BACKEND_PORT}`,
      cwd: '../backend',
      url: `http://127.0.0.1:${BACKEND_PORT}/api/v1/health`,
      timeout: 120_000,
      reuseExistingServer: false,
      stdout: 'pipe',
      stderr: 'pipe',
    },
    {
      command: `npx vite --port ${FRONTEND_PORT} --strictPort`,
      url: `http://localhost:${FRONTEND_PORT}`,
      env: { VITE_BACKEND_URL: `http://127.0.0.1:${BACKEND_PORT}` },
      timeout: 60_000,
      reuseExistingServer: false,
    },
  ],
})
