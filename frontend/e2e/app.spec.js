// End-to-end tests: a real browser drives the real app (React + FastAPI + embedding model +
// cross-encoder + ChromaDB + database). See playwright.config.js for the throwaway stack.
import path from 'node:path'
import { test, expect } from '@playwright/test'

const fixture = (name) => path.join(import.meta.dirname, 'fixtures', name)
const PASSWORD = 'e2e-password-123' // throwaway accounts in the throwaway e2e database
const uniqueEmail = (name) => `${name}-${Date.now()}-${Math.floor(Math.random() * 1e5)}@example.com`

async function register(page, email, name) {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Create account' }).click()
  if (name) await page.getByLabel(/Name/).fill(name)
  await page.getByLabel('Email').fill(email)
  await page.getByLabel(/^Password/).fill(PASSWORD)
  await page.getByRole('button', { name: 'Create account' }).click()
  await expect(page.getByRole('button', { name: 'Log out' })).toBeVisible()
}

async function login(page, email, password = PASSWORD) {
  await page.getByLabel('Email').fill(email)
  await page.getByLabel('Password').fill(password)
  await page.getByRole('button', { name: 'Log in' }).click()
}

async function upload(page, file) {
  await page.getByTestId('file-input').setInputFiles(fixture(file))
}

async function uploadAndWait(page, file, pages) {
  await upload(page, file)
  // first upload of a run also loads the embedding model, so allow extra time
  const label = pages === 1 ? "page" : "pages"
  await expect(page.getByText(`${file} is ready (${pages} ${label}).`)).toBeVisible({ timeout: 90_000 })
}

async function ask(page, question) {
  const box = page.getByLabel('Question')
  await box.fill(question)
  await box.press('Enter')
}

const answers = (page) => page.getByRole('region', { name: 'Chat' })

test('full journey: register, upload, ask with citations, follow-up, history survives reload', async ({ page }) => {
  await register(page, uniqueEmail('journey'), 'Journey Tester')
  await expect(page.getByText('All systems ready')).toBeVisible()
  await expect(page.getByText('No documents yet')).toBeVisible()
  await expect(page.getByLabel('Question')).toBeDisabled()

  await uploadAndWait(page, 'X200_Manual.pdf', 3)
  const list = page.getByRole('list', { name: 'Documents' })
  await expect(list.getByText('X200_Manual.pdf')).toBeVisible()
  await expect(list.getByText('Ready')).toBeVisible()
  await expect(list.getByText('3 pages')).toBeVisible()

  await ask(page, 'How do I reset the X200?')
  await expect(page.getByText(/Searching your documents/)).toBeVisible()
  const citation = page.getByRole('button', { name: 'Source 1: X200_Manual.pdf, page 2' })
  await expect(citation).toBeVisible({ timeout: 90_000 })
  await citation.click()
  await expect(answers(page).locator('blockquote')).toContainText('hold the red RESET button for ten seconds')

  await ask(page, 'How long is the warranty?')
  await expect(page.getByRole('button', { name: 'Source 1: X200_Manual.pdf, page 3' })).toBeVisible({ timeout: 60_000 })

  await page.getByRole('tab', { name: 'Chats (1)' }).click()
  const history = page.getByRole('list', { name: 'Chat history' })
  await expect(history.getByText('How do I reset the X200?')).toBeVisible()
  await expect(history.getByText(/2 questions/)).toBeVisible()

  // a refresh keeps the login, and the saved chat reloads from the database
  await page.reload()
  await expect(page.getByRole('button', { name: 'Log out' })).toBeVisible()
  await page.getByRole('tab', { name: 'Chats (1)' }).click()
  await page.getByRole('list', { name: 'Chat history' }).getByText('How do I reset the X200?').click()
  await expect(answers(page).getByText('How long is the warranty?')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Source 1: X200_Manual.pdf, page 2' })).toBeVisible()
})

test('bad files are rejected with clear messages and nothing is stored', async ({ page }) => {
  await register(page, uniqueEmail('rejects'))
  await upload(page, 'Corrupted.pdf')
  await expect(page.getByText('Corrupted.pdf: The PDF is corrupted or unreadable.')).toBeVisible({ timeout: 60_000 })
  await upload(page, 'Encrypted.pdf')
  await expect(page.getByText(/Encrypted\.pdf: The PDF is password-protected/)).toBeVisible()
  await upload(page, 'setup.exe')
  await expect(page.getByText('setup.exe: only PDF, DOCX and TXT files are supported.')).toBeVisible()
  await page.reload()
  await expect(page.getByText('No documents yet')).toBeVisible()
})

test('questions the documents cannot answer get "not found"', async ({ page }) => {
  await register(page, uniqueEmail('notfound'))
  await uploadAndWait(page, 'Garden_Pump.pdf', 1)
  await ask(page, 'What is the capital of France?')
  await expect(page.getByText('Not found in your documents')).toBeVisible({ timeout: 60_000 })
  await ask(page, 'What is the spare sieve part number?')
  await expect(page.getByRole('button', { name: 'Source 1: Garden_Pump.pdf, page 1' })).toBeVisible({ timeout: 60_000 })
})

test("users never see each other's documents, answers or chats", async ({ browser }) => {
  const alice = await (await browser.newContext()).newPage()
  await register(alice, uniqueEmail('alice'))
  await uploadAndWait(alice, 'X200_Manual.pdf', 3)
  await ask(alice, 'What does the warranty not cover?')
  await expect(alice.getByRole('button', { name: /Source 1: X200_Manual\.pdf/ })).toBeVisible({ timeout: 60_000 })

  const bob = await (await browser.newContext()).newPage()
  await register(bob, uniqueEmail('bob'))
  await expect(bob.getByText('No documents yet')).toBeVisible()
  await bob.getByRole('tab', { name: 'Chats' }).click()
  await expect(bob.getByText('No chats yet')).toBeVisible()
  await bob.getByRole('tab', { name: 'Documents' }).click()
  await uploadAndWait(bob, 'Garden_Pump.pdf', 1)
  await ask(bob, 'What does the X200 warranty not cover?')
  await expect(bob.getByText('Not found in your documents')).toBeVisible({ timeout: 60_000 })
  await expect(bob.getByText('X200_Manual.pdf')).toHaveCount(0)
})

test('log out, wrong password, log back in, duplicate account', async ({ page }) => {
  const email = uniqueEmail('auth')
  await register(page, email)
  await page.getByRole('button', { name: 'Log out' }).click()
  await expect(page.getByRole('button', { name: 'Log in' })).toBeVisible()

  await login(page, email, 'wrong-password-1')
  await expect(page.getByRole('alert')).toHaveText('Incorrect email or password')
  await login(page, email.toUpperCase()) // emails are case-insensitive
  await expect(page.getByRole('button', { name: 'Log out' })).toBeVisible()

  await page.getByRole('button', { name: 'Log out' }).click()
  await page.getByRole('tab', { name: 'Create account' }).click()
  await page.getByLabel('Email').fill(email)
  await page.getByLabel(/^Password/).fill(PASSWORD)
  await page.getByRole('button', { name: 'Create account' }).click()
  await expect(page.getByRole('alert')).toHaveText('An account with this email already exists')
})

test('a tampered or expired login token sends you back to the login screen', async ({ page }) => {
  await register(page, uniqueEmail('token'))
  await page.evaluate(() => localStorage.setItem('intellidoc.token', 'not-a-real-token'))
  await page.reload()
  await expect(page.getByRole('button', { name: 'Log in' })).toBeVisible()
})

test('delete a document and a chat (after confirming)', async ({ page }) => {
  page.on('dialog', (dialog) => dialog.accept())
  await register(page, uniqueEmail('delete'))
  await uploadAndWait(page, 'Garden_Pump.pdf', 1)
  await ask(page, 'What is the spare sieve part number?')
  await expect(page.getByRole('button', { name: /Source 1: Garden_Pump\.pdf/ })).toBeVisible({ timeout: 60_000 })

  await page.getByRole('tab', { name: 'Chats (1)' }).click()
  await page.getByRole('button', { name: 'Delete chat What is the spare sieve part number?' }).click()
  await expect(page.getByText('No chats yet')).toBeVisible()
  await expect(page.getByText('Ask a question about your documents')).toBeVisible()

  await page.getByRole('tab', { name: /Documents/ }).click()
  await page.getByRole('button', { name: 'Delete Garden_Pump.pdf' }).click()
  await expect(page.getByText('No documents yet')).toBeVisible()
  await page.reload()
  await expect(page.getByText('No documents yet')).toBeVisible()
})

test('works at phone width without sideways scrolling', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 })
  await register(page, uniqueEmail('mobile'))
  await expect(page.getByText(/Drag a PDF here/)).toBeVisible()
  await expect(page.getByLabel('Question')).toBeVisible()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
  expect(overflow).toBeLessThanOrEqual(0)
})
