// Captures the README screenshots (docs/screenshots/*.png) from the real app.
// Skipped unless SCREENSHOTS=1, so normal test runs and CI never touch the images:
//   $env:SCREENSHOTS="1"; $env:E2E_LLM_PROVIDER="ollama"; npx playwright test screenshots
import path from 'node:path'
import { test, expect } from '@playwright/test'

const OUT = path.join(import.meta.dirname, '..', '..', 'docs', 'screenshots')
const fixture = (name) => path.join(import.meta.dirname, 'fixtures', name)
const shot = (page, name) => page.screenshot({ path: path.join(OUT, `${name}.png`) })

test.skip(!process.env.SCREENSHOTS, 'set SCREENSHOTS=1 to capture README screenshots')
test.use({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 })

test('capture README screenshots', async ({ page }) => {
  test.setTimeout(300_000)
  await page.goto('/')
  await page.getByRole('tab', { name: 'Create account' }).click()
  await page.getByLabel(/Name/).fill('Priya')
  await page.getByLabel('Email').fill(`demo-${Date.now()}@example.com`)
  await page.getByLabel(/^Password/).fill('demo-password-123')
  await shot(page, '01-create-account')
  await page.getByRole('button', { name: 'Create account' }).click()
  await expect(page.getByRole('button', { name: 'Log out' })).toBeVisible()

  for (const [file, pages] of [['X200_Manual.pdf', 3], ['Garden_Pump.pdf', 1]]) {
    await page.getByTestId('file-input').setInputFiles(fixture(file))
    await expect(page.getByText(new RegExp(`${file.replace('.', '\\.')} is ready \\(${pages} page`))).toBeVisible({ timeout: 120_000 })
  }
  await expect(page.getByText('All systems ready')).toBeVisible({ timeout: 70_000 })

  const ask = async (question) => {
    await page.getByLabel('Question').fill(question)
    await page.getByLabel('Question').press('Enter')
  }
  await ask('What does the warranty not cover?')
  await expect(page.getByText(/Searching your documents/)).toBeVisible()
  await page.waitForTimeout(1500)
  await shot(page, '02-thinking')
  await expect(page.getByRole('button', { name: /^Source 1: X200_Manual\.pdf/ })).toBeVisible({ timeout: 240_000 })

  await ask('How do I reset the X200 and which filter does it use?')
  const cite = page.getByRole('button', { name: /^Source 1: X200_Manual\.pdf, page 2/ }).last()
  await expect(cite).toBeVisible({ timeout: 240_000 })
  await cite.click()
  await page.waitForTimeout(500)
  await shot(page, '03-answer-with-citations')

  await page.getByRole('tab', { name: /Chats/ }).click()
  await ask('What is the capital of France?')
  await expect(page.getByText('Not found in your documents')).toBeVisible({ timeout: 120_000 })
  await page.waitForTimeout(500)
  await shot(page, '04-chat-history-and-not-found')

  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('tab', { name: /Documents/ }).click()
  await page.waitForTimeout(500)
  await shot(page, '05-mobile')
})
