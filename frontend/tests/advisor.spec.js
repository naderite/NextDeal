const { test, expect } = require('@playwright/test');

test('advisor can inspect a real client, create a case, edit, approve and dispatch a manual draft', async ({ page }) => {
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Overview', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Clients', exact: true }).click();
  await page.getByRole('textbox', { name: 'Search clients' }).fill('pm:6958');
  await page.getByRole('button', { name: /Company 6958/ }).click();
  await expect(page.getByRole('heading', { name: 'Recommended next products' })).toBeVisible();
  await page.getByRole('button', { name: 'Start a conversation' }).click();
  await expect(page.getByRole('heading', { name: 'Advisor follow-up', exact: true })).toBeVisible();
  await page.getByText('Write a manual draft', { exact: true }).click();
  await page.getByLabel('Advisor draft', { exact: true }).fill('Hello, this is a manually written browser-test draft. Please consult your advisor for coverage confirmation.');
  await page.getByRole('button', { name: 'Save manual draft for approval' }).click();
  await expect(page.getByRole('textbox', { name: 'Response draft' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Simulate dispatch', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Approve', exact: true }).click();
  await page.getByRole('button', { name: 'Simulate dispatch', exact: true }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Message placed' })).toBeVisible();
  await page.getByRole('button', { name: 'Outbox', exact: true }).click();
  await expect(page.getByText('Hello, this is a manually written browser-test draft.', { exact: false })).toBeVisible();
  expect(errors).toEqual([]);
});

test('data governance UI quarantines a broken batch and exposes results', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Data & quality', exact: true }).click();
  await page.getByRole('textbox', { name: 'Batch records JSON' }).fill(JSON.stringify([{ id: 'browser-bad', kind: 'corporate', label: '', age: -5 }]));
  await page.getByRole('button', { name: 'Check and import', exact: true }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Batch checked' })).toBeVisible();
  await expect(page.getByText(/quarantined · 2 findings/)).toBeVisible();
  await page.getByRole('button', { name: 'Evaluation', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Objective evidence register' })).toBeVisible();
  await expect(page.getByRole('cell', { name: /Identity deduplication/ })).toBeVisible();
});

test('mobile navigation remains usable without horizontal page overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.getByRole('button', { name: 'Clients', exact: true }).click();
  await expect(page.getByRole('textbox', { name: 'Search clients' })).toBeVisible();
  const fits = await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1);
  expect(fits).toBeTruthy();
});
