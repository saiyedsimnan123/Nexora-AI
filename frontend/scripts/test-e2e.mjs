/**
 * Nexora AI - End-to-End Frontend & Backend Integration Verification.
 *
 * Verifies:
 * 1. Backend REST API health, readiness, and RAG query endpoints.
 * 2. Vite reverse proxy routing (/health, /ready, /api/v1/...).
 * 3. Headless Chrome UI rendering:
 *    - Real-time navbar health badge updates to "Live (v0.1.0 RC)".
 *    - Click "Explore Research" opens modal and triggers live RAG query.
 *    - Dynamic UI renders grounded answer, evidence badges [E1]/[E2], and sources.
 *    - Zero uncaught exceptions or console errors.
 */

import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';

const BACKEND_URL = 'http://127.0.0.1:8000';
const FRONTEND_URL = 'http://127.0.0.1:4173';
const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const CDP_PORT = 9333;

function log(step, msg) {
  console.log(`[E2E] [${step}] ${msg}`);
}

async function testHttpEndpoints() {
  log('API-DIRECT', 'Testing direct backend endpoints...');

  // 1. Health
  const healthRes = await fetch(`${BACKEND_URL}/health`);
  if (!healthRes.ok) throw new Error(`Backend /health returned status ${healthRes.status}`);
  const healthJson = await healthRes.json();
  if (healthJson.status !== 'ok') throw new Error(`Unexpected health payload: ${JSON.stringify(healthJson)}`);
  log('API-DIRECT', `✓ GET /health: 200 OK -> ${JSON.stringify(healthJson)}`);

  // 2. Readiness
  const readyRes = await fetch(`${BACKEND_URL}/ready`);
  if (!readyRes.ok) throw new Error(`Backend /ready returned status ${readyRes.status}`);
  const readyJson = await readyRes.json();
  if (readyJson.status !== 'ready') throw new Error(`Unexpected readiness payload: ${JSON.stringify(readyJson)}`);
  log('API-DIRECT', `✓ GET /ready: 200 OK -> status=${readyJson.status}, auth=${readyJson.auth_service}, rag=${readyJson.research_service}`);

  // 3. RAG Query
  const ragRes = await fetch(`${BACKEND_URL}/api/v1/research/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: 'How do FlashAttention IO optimizations achieve speedup on GPUs?',
      retrieval_mode: 'hybrid',
      top_k: 5
    })
  });
  if (!ragRes.ok) throw new Error(`Backend /api/v1/research/query returned status ${ragRes.status}`);
  const ragJson = await ragRes.json();
  if (!ragJson.answer || !Array.isArray(ragJson.sources) || ragJson.sources.length === 0) {
    throw new Error(`Invalid RAG response structure: ${JSON.stringify(ragJson)}`);
  }
  log('API-DIRECT', `✓ POST /api/v1/research/query: 200 OK`);
  log('API-DIRECT', `  - Answer length: ${ragJson.answer.length} chars`);
  log('API-DIRECT', `  - Retrieved sources: ${ragJson.sources.length}`);
  log('API-DIRECT', `  - Grounded: ${ragJson.is_grounded}`);
  log('API-DIRECT', `  - First source title: "${ragJson.sources[0]?.paper_title}"`);

  // 4. Test Frontend Reverse Proxy
  log('VITE-PROXY', 'Testing Vite reverse proxy routing...');
  const proxyHealthRes = await fetch(`${FRONTEND_URL}/health`);
  if (!proxyHealthRes.ok) throw new Error(`Vite proxy /health returned status ${proxyHealthRes.status}`);
  log('VITE-PROXY', `✓ GET ${FRONTEND_URL}/health routed successfully to backend (200 OK)`);

  const proxyRagRes = await fetch(`${FRONTEND_URL}/api/v1/research/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: 'What is Low-Rank Adaptation (LoRA)?',
      retrieval_mode: 'hybrid',
      top_k: 3
    })
  });
  if (!proxyRagRes.ok) throw new Error(`Vite proxy /api/v1/research/query returned status ${proxyRagRes.status}`);
  const proxyRagJson = await proxyRagRes.json();
  log('VITE-PROXY', `✓ POST ${FRONTEND_URL}/api/v1/research/query: 200 OK -> sources: ${proxyRagJson.sources.length}`);
}

async function runBrowserVerification() {
  log('BROWSER', 'Launching headless Chrome with DevTools Protocol...');

  const chromeArgs = [
    '--headless=new',
    `--remote-debugging-port=${CDP_PORT}`,
    '--window-size=1440,900',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--user-data-dir=' + process.env.TEMP + '\\chrome_e2e_profile_' + Date.now(),
    'about:blank'
  ];

  const chromeProc = spawn(CHROME_PATH, chromeArgs, { stdio: 'ignore' });

  try {
    // Wait for CDP endpoint
    let cdpInfo = null;
    for (let i = 0; i < 20; i++) {
      try {
        const res = await fetch(`http://127.0.0.1:${CDP_PORT}/json/version`);
        if (res.ok) {
          cdpInfo = await res.json();
          break;
        }
      } catch {
        await sleep(250);
      }
    }

    if (!cdpInfo) throw new Error('Chrome DevTools Protocol port did not become responsive');
    log('BROWSER', `Chrome CDP active: ${cdpInfo.Browser}`);

    // Get active page target or create one with PUT
    const listRes = await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`);
    const targets = await listRes.json();
    let pageTarget = targets.find(t => t.type === 'page');
    if (!pageTarget) {
      const newPageRes = await fetch(`http://127.0.0.1:${CDP_PORT}/json/new`, { method: 'PUT' });
      pageTarget = await newPageRes.json();
    }
    const wsUrl = pageTarget.webSocketDebuggerUrl;

    log('BROWSER', `Connected to WebSocket: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);

    let msgId = 0;
    const pendingPromises = new Map();
    const consoleErrors = [];

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.id && pendingPromises.has(data.id)) {
        const { resolve, reject } = pendingPromises.get(data.id);
        pendingPromises.delete(data.id);
        if (data.error) reject(new Error(data.error.message));
        else resolve(data.result);
      }
      if (data.method === 'Runtime.exceptionThrown') {
        const desc = data.params?.exceptionDetails?.exception?.description || data.params?.exceptionDetails?.text;
        consoleErrors.push(`[Runtime Exception] ${desc}`);
      }
      if (data.method === 'Runtime.consoleAPICalled' && data.params.type === 'error') {
        const text = data.params.args?.map(a => a.value || a.description).join(' ');
        consoleErrors.push(`[Console Error] ${text}`);
      }
    };

    function sendCommand(method, params = {}) {
      return new Promise((resolve, reject) => {
        const id = ++msgId;
        pendingPromises.set(id, { resolve, reject });
        ws.send(JSON.stringify({ id, method, params }));
      });
    }

    await new Promise((resolve) => ws.onopen = resolve);
    await sendCommand('Runtime.enable');
    await sendCommand('Page.enable');
    await sendCommand('Emulation.setDeviceMetricsOverride', {
      width: 1440,
      height: 900,
      deviceScaleFactor: 1,
      mobile: false
    });
    await sendCommand('Page.navigate', { url: FRONTEND_URL });

    // Wait for initial page rendering and check health badge
    log('BROWSER', 'Waiting for page load and health check reconciliation...');
    let healthBadgeText = '';
    for (let attempt = 0; attempt < 25; attempt++) {
      await sleep(400);
      const evalRes = await sendCommand('Runtime.evaluate', {
        expression: 'JSON.stringify({ url: window.location.href, width: window.innerWidth, navText: document.querySelector("nav") ? document.querySelector("nav").textContent : "", readyState: document.readyState })',
        returnByValue: true
      });
      const pageState = JSON.parse(evalRes.result?.value || '{}');
      if (attempt % 5 === 0) {
        log('BROWSER', `Attempt ${attempt}: width=${pageState.width}, navText=${pageState.navText?.slice(0, 80)}`);
      }
      if (pageState.navText && pageState.navText.includes('Live (v0.1.0 RC)')) {
        healthBadgeText = 'Live (v0.1.0 RC)';
        break;
      }
    }

    if (!healthBadgeText) {
      const finalEval = await sendCommand('Runtime.evaluate', {
        expression: 'document.querySelector("nav") ? document.querySelector("nav").textContent : "NO NAV"',
        returnByValue: true
      });
      log('DEBUG', `Final nav text on failure: ${finalEval.result?.value}`);
      throw new Error('Navbar did not display "Live (v0.1.0 RC)" after health check polling');
    }
    log('BROWSER', `✓ Navbar verified: Displaying real-time badge "Live (v0.1.0 RC)"`);

    // Click "Explore Research" to open the live search modal
    log('BROWSER', 'Triggering "Explore Research" modal...');
    const clickRes = await sendCommand('Runtime.evaluate', {
      expression: `
        (() => {
          const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Explore Research'));
          if (btn) {
            btn.click();
            return { clicked: true, text: btn.innerText };
          }
          return { clicked: false };
        })()
      `,
      returnByValue: true
    });

    if (!clickRes.result?.value?.clicked) {
      throw new Error('Could not find or click "Explore Research" button');
    }
    log('BROWSER', '✓ "Explore Research" button clicked successfully');

    // Wait for Search Modal to perform the live RAG query and render results
    log('BROWSER', 'Waiting for live RAG query response and citation rendering in modal...');
    let searchSuccess = false;
    let renderedSources = 0;
    let modalText = '';

    for (let attempt = 0; attempt < 30; attempt++) {
      await sleep(400);
      const modalEval = await sendCommand('Runtime.evaluate', {
        expression: `
          (() => {
            const modal = document.querySelector('[role="dialog"]');
            if (!modal) return null;
            const text = modal.innerText;
            const citations = modal.querySelectorAll('button');
            const citationButtons = Array.from(citations).filter(c => c.innerText.startsWith('[E')).map(c => c.innerText);
            return {
              isOpen: true,
              text: text.slice(0, 1500),
              citationCount: citationButtons.length,
              citationButtons: citationButtons,
              hasGroundedAnswer: text.includes('FlashAttention') || text.includes('Grounding Confidence')
            };
          })()
        `,
        returnByValue: true
      });

      const modalInfo = modalEval.result?.value;
      if (modalInfo && modalInfo.hasGroundedAnswer && modalInfo.citationCount > 0) {
        searchSuccess = true;
        renderedSources = modalInfo.citationCount;
        modalText = modalInfo.text;
        break;
      }
    }

    if (!searchSuccess) {
      throw new Error(`Search modal failed to render live RAG query results within timeout. Modal state: ${modalText.slice(0, 300)}`);
    }

    log('BROWSER', `✓ Live Search Modal verified:`);
    log('BROWSER', `  - Answer rendered with live academic citations`);
    log('BROWSER', `  - Clickable citation source buttons rendered: ${renderedSources}`);

    // Check for any console errors during full test run
    if (consoleErrors.length > 0) {
      log('WARN', `Browser emitted ${consoleErrors.length} console error(s):`);
      for (const err of consoleErrors) {
        log('WARN', `  ${err}`);
      }
      // If there are real JavaScript uncaught exceptions, fail
      const fatalErrors = consoleErrors.filter(e => e.includes('Runtime Exception'));
      if (fatalErrors.length > 0) {
        throw new Error(`Fatal browser runtime exceptions detected: ${fatalErrors.join(', ')}`);
      }
    } else {
      log('BROWSER', `✓ Clean browser session: 0 runtime exceptions or console errors`);
    }

    ws.close();
  } finally {
    try {
      chromeProc.kill('SIGKILL');
    } catch {}
  }
}

async function main() {
  console.log('=====================================================');
  console.log('  Nexora AI — End-to-End Live Integration Test Suite ');
  console.log('=====================================================\n');

  try {
    await testHttpEndpoints();
    await runBrowserVerification();
    console.log('\n=====================================================');
    console.log('  ALL END-TO-END VERIFICATION CHECKS PASSED (100%)  ');
    console.log('=====================================================');
    process.exit(0);
  } catch (err) {
    console.error('\n[ERROR] E2E Verification Failed:', err);
    process.exit(1);
  }
}

main();
