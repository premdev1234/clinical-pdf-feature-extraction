const puppeteer = require("puppeteer-extra");
const StealthPlugin = require("puppeteer-extra-plugin-stealth");
const fs = require("fs");
const path = require("path");
const readline = require("readline");
const csv = require("csv-parser");
const UserAgent = require("user-agents");

puppeteer.use(StealthPlugin());

/* ------------------- CONFIG ------------------- */
const concurrency = 4;
const csvFilePath = path.join(__dirname, "spinocerebellar_ataxias.csv");
const proxyList = [];

const maxPagesPerQuery = 50;
const outDir = path.join(__dirname, "data", "json");
const debugDir = path.join(__dirname, "debug");
fs.mkdirSync(outDir, { recursive: true });
fs.mkdirSync(debugDir, { recursive: true });

/* ------------------- HELPERS ------------------- */
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const randomInt = (a, b) => Math.floor(a + Math.random() * (b - a));
function log(...args) { console.log(new Date().toISOString(), ...args); }

function askEnter(promptText = "Press ENTER to continue...") {
  return new Promise((resolve) => {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    rl.question(promptText, () => { rl.close(); resolve(); });
  });
}

/* ------------------- CSV LOADING ------------------- */
async function loadQueriesFromCSV(filePath) {
  return new Promise((resolve, reject) => {
    const results = [];
    if (!fs.existsSync(filePath)) {
      return resolve([]);
    }
    fs.createReadStream(filePath)
      .pipe(csv())
      .on("data", (row) => {
        const diseaseName = row["HPO_Disease_Name"] || row["HPO_Disease_Name "];
        if (diseaseName && diseaseName.trim().length > 0) results.push(diseaseName.trim());
      })
      .on("end", () => {
        log(`Loaded ${results.length} queries from CSV.`);
        resolve(results);
      })
      .on("error", (err) => reject(err));
  });
}

/* ------------------- CAPTCHA ------------------- */
async function isCaptchaPresent(page) {
  try {
    const bodyText = await page.evaluate(() => document.body.innerText || "");
    if (/unusual traffic|verify you're human|recaptcha|please show|not a robot/i.test(bodyText)) return true;
    const iframe = await page.$('iframe[src*="recaptcha"], iframe[src*="captcha"]');
    if (iframe) return true;
    return false;
  } catch (err) {
    log("isCaptchaPresent error:", err.message);
    return false;
  }
}

async function handleCaptcha(page, contextTag) {
  const ts = Date.now();
  const ssPath = path.join(debugDir, `captcha_${contextTag}_${ts}.png`);
  const htmlPath = path.join(debugDir, `captcha_${contextTag}_${ts}.html`);
  const cookiePath = path.join(debugDir, `cookies_${contextTag}_${ts}.json`);
  try {
    log(`[${contextTag}] CAPTCHA detected. Saving screenshot + HTML + cookies.`);
    await page.screenshot({ path: ssPath, fullPage: true });
    fs.writeFileSync(htmlPath, await page.content());
    fs.writeFileSync(cookiePath, JSON.stringify(await page.cookies(), null, 2));
    log(`[${contextTag}] Saved ${ssPath} and ${htmlPath}. Please solve manually in the browser.`);
    await askEnter(`[${contextTag}] Press ENTER after solving the CAPTCHA in the browser to continue...`);
    log(`[${contextTag}] Resuming after manual solve.`);
    await page.reload({ waitUntil: "domcontentloaded" });
    await sleep(1500);
  } catch (e) {
    log(`[${contextTag}] handleCaptcha error: ${e.message}`);
  }
}

/* ------------------- Extraction ------------------- */
async function extractStructured(page) {
  return await page.evaluate(() => {
    const nodes = Array.from(document.querySelectorAll(".gs_ri"));
    return nodes.map(el => {
      const titleEl = el.querySelector("h3 a");
      const title = titleEl ? titleEl.textContent.trim() : null;
      const url = titleEl ? titleEl.href : null;
      const authors = el.querySelector(".gs_a") ? el.querySelector(".gs_a").innerText.trim() : null;
      const snippet = el.querySelector(".gs_rs") ? el.querySelector(".gs_rs").innerText.trim() : null;
      return { title, url, authors, snippet };
    });
  });
}

/* ------------------- Cookies ------------------- */
async function saveCookies(page, filename) {
  try {
    const cookies = await page.cookies();
    fs.writeFileSync(filename, JSON.stringify(cookies, null, 2));
  } catch (err) {
    log("saveCookies error:", err.message);
  }
}
async function loadCookies(page, filename) {
  try {
    if (!fs.existsSync(filename)) return;
    const cookies = JSON.parse(fs.readFileSync(filename, "utf8"));
    if (Array.isArray(cookies) && cookies.length) await page.setCookie(...cookies);
  } catch (err) {
    log("loadCookies error:", err.message);
  }
}

/* ------------------- Human-like interactions ------------------- */
async function smallHumanMouseMove(page, distance = 12, steps = 6) {
  try {
    const vw = await page.evaluate(() => ({ w: window.innerWidth, h: window.innerHeight }));
    const startX = Math.floor(vw.w * (0.45 + Math.random() * 0.1));
    const startY = Math.floor(vw.h * (0.45 + Math.random() * 0.1));
    const dx = (Math.random() > 0.5 ? 1 : -1) * distance;
    const dy = (Math.random() > 0.5 ? 1 : -1) * Math.floor(distance / 3);
    const microDelayMs = 3;
    for (let i = 0; i <= steps; i++) {
      const x = Math.round(startX + (dx * i) / steps);
      const y = Math.round(startY + (dy * i) / steps);
      await page.mouse.move(x, y);
      await sleep(microDelayMs);
    }
    await sleep(randomInt(30, 140));
  } catch (err) {
    log("mouse move failed:", err.message);
  }
}

async function smallScroll(page) {
  try {
    for (let i = 0; i < randomInt(1, 3); i++) {
      await page.evaluate((amt) => window.scrollBy(0, amt), randomInt(200, 600));
      await sleep(randomInt(300, 900));
    }
    if (Math.random() < 0.2) await page.evaluate(() => window.scrollTo(0, 0));
  } catch (e) {}
}

/* ------------------- Worker ------------------- */
async function createWorker(workerId, proxy, jobQueue) {
  const ua = new UserAgent().toString();

  const launchArgs = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--window-size=1280,900",
  ];
  if (proxy) launchArgs.push(`--proxy-server=${proxy}`);

  const browser = await puppeteer.launch({
    headless: false,
    args: launchArgs,
    defaultViewport: null,
  });

  const page = await browser.newPage();
  await page.setUserAgent(ua);

  if (proxy && proxy.includes("@")) {
    try {
      const [creds, hostPart] = proxy.split("@");
      const [username, password] = creds.split(":");
      await page.authenticate({ username, password });
    } catch (e) {
      log(`[w${workerId}] proxy auth parse error: ${e.message}`);
    }
  }

  log(`[w${workerId}] started (proxy=${proxy || "none"}) UA=${ua}`);

  let lastRequestTime = 0;
  const minIntervalMs = 8000 + randomInt(0, 4000);
  let backoff = 0;

  while (jobQueue.length > 0) {
    const keyword = jobQueue.shift();
    if (!keyword) break;

    const outFile = path.join(outDir, `${keyword.replace(/\s+/g, "_")}.json`);
    if (fs.existsSync(outFile)) {
      log(`[w${workerId}] skipping already-scraped "${keyword}"`);
      continue;
    }

    log(`[w${workerId}] processing "${keyword}"`);

    const cookieFile = path.join(debugDir, `${keyword.replace(/\s+/g, "_")}_cookies.json`);
    const results = [];

    try {
      await loadCookies(page, cookieFile);

      for (let pageIndex = 0; pageIndex < maxPagesPerQuery; pageIndex++) {
        const now = Date.now();
        const elapsed = now - lastRequestTime;
        if (elapsed < minIntervalMs) {
          await sleep(minIntervalMs - elapsed + randomInt(200, 800));
        }

        const start = pageIndex * 10;
        const url = `https://scholar.google.com/scholar?start=${start}&q=${encodeURIComponent(keyword)}&hl=en&as_sdt=0,5`;
        log(`[w${workerId}] [${keyword}] page ${pageIndex + 1} -> ${url}`);

        await smallHumanMouseMove(page);
        await sleep(randomInt(200, 700));

        try {
          await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60000 });
        } catch (navErr) {
          log(`[w${workerId}] navigation error (page ${pageIndex + 1}): ${navErr.message}`);
          backoff++;
          const sleepMs = Math.min(30000, 2000 * Math.pow(2, Math.min(backoff, 6)));
          log(`[w${workerId}] backing off for ${sleepMs}ms (backoff=${backoff})`);
          await sleep(sleepMs);
          try { await page.reload({ waitUntil: "domcontentloaded", timeout: 60000 }); } catch {}
          if (backoff > 5) {
            log(`[w${workerId}] too many consecutive failures, aborting keyword "${keyword}"`);
            break;
          } else {
            pageIndex--;
            continue;
          }
        }

        lastRequestTime = Date.now();
        backoff = 0;

        await sleep(randomInt(900, 2000));
        await smallScroll(page);
        await smallHumanMouseMove(page, 8, 8);

        if (await isCaptchaPresent(page)) {
          await handleCaptcha(page, `w${workerId}_${keyword}_p${pageIndex + 1}`);
          await saveCookies(page, cookieFile);
        }

        const pageResults = await extractStructured(page);
        log(`[w${workerId}] [${keyword}] extracted ${pageResults.length} items from page ${pageIndex + 1}`);
        if (!pageResults.length) {
          log(`[w${workerId}] [${keyword}] no results on page ${pageIndex + 1}; assuming end of results.`);
          break;
        }
        results.push(...pageResults);
        await saveCookies(page, cookieFile);
        await sleep(randomInt(700, 1600));
      }

      const seen = new Set();
      const uniqueResults = [];
      for (const r of results) {
        const key = r.url || r.title;
        if (!key) continue;
        if (!seen.has(key)) {
          seen.add(key);
          uniqueResults.push(r);
        }
      }

      const outFileFinal = path.join(outDir, `${keyword.replace(/\s+/g, "_")}.json`);
      fs.writeFileSync(outFileFinal, JSON.stringify(uniqueResults, null, 2));
      log(`[w${workerId}] wrote ${uniqueResults.length} unique results to ${outFileFinal}`);
    } catch (err) {
      log(`[w${workerId}] error processing "${keyword}": ${err.message}`);
    }

    await sleep(randomInt(1000, 3000));
  }

  try { await page.close(); } catch (e) {}
  try { await browser.close(); } catch (e) {}
  log(`[w${workerId}] finished.`);
}

/* ------------------- MAIN ------------------- */
(async () => {
  const queries = await loadQueriesFromCSV(csvFilePath);
  if (!queries || queries.length === 0) {
    log("No queries found in CSV. Exiting.");
    process.exit(1);
  }

  const cleaned = Array.from(new Set(queries.map(q => q.trim()).filter(Boolean)));
  log(`Total unique queries to process: ${cleaned.length}`);

  const jobQueue = cleaned.slice();
  const proxies = proxyList.slice();
  const workers = [];

  for (let i = 0; i < concurrency; i++) {
    const proxy = proxies.length ? proxies[i % proxies.length] : null;
    const w = createWorker(i + 1, proxy, jobQueue);
    workers.push(w);
    await sleep(randomInt(400, 1200));
  }
  await Promise.all(workers);
  log("All workers done.");
})();
