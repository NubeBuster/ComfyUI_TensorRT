/**
 * TensorRT Loader Auto — Progress Toast
 *
 * Listens for `trt_auto_progress` WebSocket events and shows a floating
 * toast with stage text and progress bar during engine build/load/refit.
 *
 * Phases: searching -> building -> loading -> refitting -> done
 */

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const TOAST_CLASS = "trt-auto-toast";
const AUTO_DISMISS_MS = 3000;
const POS_STORAGE_KEY = "trt_auto_toast_pos";

let activeToast = null;
let dismissTimer = null;
let tickInterval = null;
let phaseStartTime = null;  // Date.now() when current phase started
let currentEta = null;      // ETA in seconds from backend
let persistent = false;     // user interacted — don't auto-dismiss

function injectStyles() {
    if (document.getElementById("trt-auto-toast-css")) return;
    const style = document.createElement("style");
    style.id = "trt-auto-toast-css";
    style.textContent = `
        .${TOAST_CLASS} {
            position: fixed;
            z-index: 99999;
            background: #1e1e1e;
            border: 1px solid #555;
            border-radius: 8px;
            padding: 12px 16px;
            min-width: 260px;
            max-width: 340px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.5);
            font-family: system-ui, -apple-system, sans-serif;
            color: #ddd;
            transition: opacity 0.3s ease;
        }
        .${TOAST_CLASS}.hiding {
            opacity: 0;
        }
        .${TOAST_CLASS} .toast-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            cursor: grab;
            user-select: none;
        }
        .${TOAST_CLASS} .toast-header.dragging {
            cursor: grabbing;
        }
        .${TOAST_CLASS} .toast-title {
            font-size: 12px;
            font-weight: 600;
            color: #76B900;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .${TOAST_CLASS} .toast-action-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: none;
            border: 1px solid #4A9EFF44;
            border-radius: 4px;
            color: #4A9EFF;
            cursor: grab;
            width: 22px;
            height: 22px;
            font-size: 13px;
            padding: 0;
            line-height: 1;
        }
        .${TOAST_CLASS} .toast-action-btn:hover {
            background: #4A9EFF22;
        }
        .${TOAST_CLASS} .toast-close {
            display: none;
        }
        .${TOAST_CLASS}.persistent .toast-action-btn {
            display: none;
        }
        .${TOAST_CLASS}.persistent .toast-close {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #5a1a1a;
            border: 1px solid #a33;
            border-radius: 4px;
            color: #e88;
            cursor: pointer;
            width: 22px;
            height: 22px;
            font-size: 14px;
            padding: 0;
            line-height: 1;
        }
        .${TOAST_CLASS}.persistent .toast-close:hover {
            background: #7a2a2a;
            color: #faa;
        }
        .${TOAST_CLASS} .toast-tooltip {
            display: none;
            position: absolute;
            bottom: calc(100% + 6px);
            right: 0;
            background: #111;
            color: #bbb;
            font-size: 11px;
            padding: 6px 10px;
            border-radius: 4px;
            border: 1px solid #444;
            white-space: nowrap;
            z-index: 100000;
            pointer-events: none;
            max-width: 280px;
            white-space: normal;
            line-height: 1.4;
        }
        .${TOAST_CLASS} .toast-close:hover + .toast-tooltip,
        .${TOAST_CLASS} .toast-tooltip.show {
            display: block;
        }
        .${TOAST_CLASS} .toast-stage {
            font-size: 13px;
            margin-bottom: 8px;
            color: #ccc;
        }
        .${TOAST_CLASS} .toast-bar-track {
            height: 6px;
            background: #333;
            border-radius: 3px;
            overflow: hidden;
        }
        .${TOAST_CLASS} .toast-bar-fill {
            height: 100%;
            background: #76B900;
            border-radius: 3px;
            transition: width 0.4s ease;
        }
        .${TOAST_CLASS} .toast-timer {
            font-size: 11px;
            color: #888;
            margin-top: 6px;
            display: flex;
            justify-content: space-between;
        }
        .${TOAST_CLASS} .toast-eta {
            cursor: help;
        }
        .${TOAST_CLASS} .toast-build-detail {
            font-size: 11px;
            color: #999;
            margin-top: 2px;
            font-family: monospace;
        }
        .${TOAST_CLASS}.done .toast-title {
            color: #4CAF50;
        }
        .${TOAST_CLASS}.done .toast-bar-fill {
            background: #4CAF50;
        }
        .${TOAST_CLASS}.done-summary {
            opacity: 0.80;
            background: #1a2e1a;
            border-color: #4CAF50;
        }
        .${TOAST_CLASS} .toast-summary {
            font-size: 11px;
            color: #ccc;
            margin-top: 6px;
            line-height: 1.5;
        }
        .${TOAST_CLASS} .eta-faster { color: #81C784; }
        .${TOAST_CLASS} .eta-slower-ok { color: #81C784; }
        .${TOAST_CLASS} .eta-slower-warn { color: #FFB74D; }
    `;
    document.head.appendChild(style);
}

function loadPosition() {
    try {
        const raw = localStorage.getItem(POS_STORAGE_KEY);
        if (!raw) return null;
        const pos = JSON.parse(raw);
        if (typeof pos.bottom !== "number" || typeof pos.right !== "number") return null;
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        if (pos.bottom < 0 || pos.bottom > vh - 40) return null;
        if (pos.right < 0 || pos.right > vw - 100) return null;
        return pos;
    } catch {
        return null;
    }
}

function savePosition(bottom, right) {
    try {
        localStorage.setItem(POS_STORAGE_KEY, JSON.stringify({ bottom, right }));
    } catch { /* ignore */ }
}

function getPosition() {
    return loadPosition() || { bottom: 80, right: 20 };
}

function positionToast(el) {
    const pos = getPosition();
    el.style.bottom = `${pos.bottom}px`;
    el.style.right = `${pos.right}px`;
    el.style.top = "auto";
    el.style.left = "auto";
}

function setupDrag(toast) {
    const header = toast.querySelector(".toast-header");
    let dragging = false;
    let startX, startY, startBottom, startRight;

    header.addEventListener("mousedown", (e) => {
        if (e.target.closest(".toast-close") || e.target.closest(".toast-tooltip")) return;
        dragging = true;
        const rect = toast.getBoundingClientRect();
        startX = e.clientX;
        startY = e.clientY;
        startBottom = window.innerHeight - rect.bottom;
        startRight = window.innerWidth - rect.right;
        header.classList.add("dragging");
        e.preventDefault();
    });

    document.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const rect = toast.getBoundingClientRect();
        const pad = 8;
        let newRight = startRight - (e.clientX - startX);
        let newBottom = startBottom - (e.clientY - startY);
        // Clamp: keep toast within viewport, pad inward from edges
        const maxRight = vw - rect.width - pad;
        const maxBottom = vh - rect.height - pad;
        newRight = Math.max(pad, Math.min(newRight, maxRight));
        newBottom = Math.max(pad, Math.min(newBottom, maxBottom));
        toast.style.bottom = `${newBottom}px`;
        toast.style.right = `${newRight}px`;
        toast.style.top = "auto";
        toast.style.left = "auto";
    });

    document.addEventListener("mouseup", () => {
        if (!dragging) return;
        dragging = false;
        header.classList.remove("dragging");
        makePersistent();
        const rect = toast.getBoundingClientRect();
        savePosition(
            window.innerHeight - rect.bottom,
            window.innerWidth - rect.right
        );
    });
}

function createToast() {
    injectStyles();
    const el = document.createElement("div");
    el.className = TOAST_CLASS;
    el.innerHTML = `
        <div class="toast-header">
            <span class="toast-title">TensorRT</span>
            <span class="toast-action-btn">\u2725</span>
            <button class="toast-close">\u00d7</button>
            <span class="toast-tooltip">Pinned because you interacted. Click to unpin (will auto-dismiss when done).</span>
        </div>
        <div class="toast-stage">Preparing\u2026</div>
        <div class="toast-build-detail" style="display: none"></div>
        <div class="toast-bar-track">
            <div class="toast-bar-fill" style="width: 0%"></div>
        </div>
        <div class="toast-timer" style="display: none">
            <span class="toast-elapsed"></span>
            <span class="toast-eta" title="Estimated from recent operations on this machine. Actual time varies with GPU load and model complexity."></span>
        </div>
    `;
    el.querySelector(".toast-close").addEventListener("click", () => {
        if (activeToast && activeToast.classList.contains("done")) {
            // Terminal state — actually dismiss
            dismiss();
        } else {
            // Active operation — just unstick, let auto-dismiss handle it later
            persistent = false;
            if (activeToast) activeToast.classList.remove("persistent");
        }
    });
    // Any user interaction makes the toast persistent
    el.addEventListener("click", (e) => {
        if (!e.target.closest(".toast-close")) makePersistent();
    });
    el.addEventListener("contextmenu", () => makePersistent());

    // Fast tooltip (100ms) for close button
    const closeBtn = el.querySelector(".toast-close");
    const tooltip = el.querySelector(".toast-tooltip");
    let tooltipTimer = null;
    closeBtn.addEventListener("mouseenter", () => {
        tooltipTimer = setTimeout(() => tooltip.classList.add("show"), 100);
    });
    closeBtn.addEventListener("mouseleave", () => {
        clearTimeout(tooltipTimer);
        tooltip.classList.remove("show");
    });

    document.body.appendChild(el);
    setupDrag(el);
    positionToast(el);
    return el;
}

function getToast() {
    if (activeToast) {
        activeToast.classList.remove("hiding", "done", "done-summary");
        return activeToast;
    }
    persistent = false;
    activeToast = createToast();
    return activeToast;
}

function formatDuration(seconds) {
    seconds = Math.round(seconds);
    if (seconds < 60) return `${seconds}s`;
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

function startTick() {
    stopTick();
    phaseStartTime = Date.now();
    console.log("[TRT tick] startTick called, currentEta=%s", currentEta);
    tickInterval = setInterval(() => {
        if (!activeToast) { stopTick(); return; }
        const elapsed = (Date.now() - phaseStartTime) / 1000;
        const timerEl = activeToast.querySelector(".toast-timer");
        const elapsedEl = activeToast.querySelector(".toast-elapsed");
        const etaEl = activeToast.querySelector(".toast-eta");
        if (!timerEl) return;

        timerEl.style.display = "flex";
        elapsedEl.textContent = `Elapsed: ${formatDuration(elapsed)}`;

        if (currentEta != null && currentEta > 0) {
            const remaining = Math.max(0, currentEta - elapsed);
            etaEl.textContent = remaining > 0 ? `ETA: ~${formatDuration(remaining)}` : `+${formatDuration(elapsed - currentEta)} over`;
            // Update progress bar based on elapsed/eta
            const pct = Math.min(95, (elapsed / currentEta) * 100);
            const fill = activeToast.querySelector(".toast-bar-fill");
            if (fill) fill.style.width = `${pct}%`;
        } else {
            if (elapsed < 2) console.log("[TRT tick] no ETA (currentEta=%s)", currentEta);
            etaEl.textContent = "";
        }
    }, 1000);
}

function stopTick() {
    if (tickInterval) { clearInterval(tickInterval); tickInterval = null; }
    phaseStartTime = null;
}

function makePersistent() {
    if (persistent || !activeToast) return;
    persistent = true;
    activeToast.classList.add("persistent");
    if (dismissTimer) {
        clearTimeout(dismissTimer);
        dismissTimer = null;
    }
}

function dismiss() {
    if (!activeToast) return;
    persistent = false;
    stopTick();
    currentEta = null;
    activeToast.classList.add("hiding");
    const el = activeToast;
    setTimeout(() => {
        el.remove();
        if (activeToast === el) activeToast = null;
    }, 300);
    if (dismissTimer) {
        clearTimeout(dismissTimer);
        dismissTimer = null;
    }
}

// Phase colors: building=blue, refitting=NVIDIA green, done=green, default=NVIDIA green
const PHASE_COLORS = {
    searching:            "#76B900",
    building:             "#4A9EFF",
    loading:              "#76B900",
    "loading cached refit": "#76B900",
    refitting:            "#76B900",
    done:                 "#4CAF50",
    cached:               "#4CAF50",
};

const PHASES = {
    searching:              { text: "Searching for engine\u2026",              pct: 5  },
    building:               { text: "Building TRT engine\u2026",               pct: 10 },
    loading:                { text: "Loading engine\u2026",                     pct: 70 },
    "loading cached refit": { text: "Loading cached refit\u2026",              pct: 70 },
    refitting:              { text: "Refitting LoRA weights\u2026",             pct: 80 },
    done:                   { text: "Done",                                     pct: 100 },
    cached:                 { text: "Engine cached \u2014 skipped",             pct: 100 },
};

// --- Disk eviction toast (persistent, separate from progress toast) ---

const EVICTION_TOAST_CLASS = "trt-eviction-toast";
const EVICTION_POS_KEY = "trt_eviction_toast_pos";

let activeEvictionToast = null;

function injectEvictionStyles() {
    if (document.getElementById("trt-eviction-toast-css")) return;
    const style = document.createElement("style");
    style.id = "trt-eviction-toast-css";
    style.textContent = `
        .${EVICTION_TOAST_CLASS} {
            position: fixed;
            z-index: 99999;
            background: #1e1e1e;
            border: 1px solid #c0a000;
            border-radius: 8px;
            padding: 12px 16px;
            min-width: 260px;
            max-width: 380px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.5);
            font-family: system-ui, -apple-system, sans-serif;
            color: #ddd;
            transition: opacity 0.3s ease;
        }
        .${EVICTION_TOAST_CLASS}.hiding { opacity: 0; }
        .${EVICTION_TOAST_CLASS} .ev-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            cursor: grab;
            user-select: none;
        }
        .${EVICTION_TOAST_CLASS} .ev-header.dragging { cursor: grabbing; }
        .${EVICTION_TOAST_CLASS} .ev-title {
            font-size: 12px;
            font-weight: 600;
            color: #c0a000;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .${EVICTION_TOAST_CLASS} .ev-close {
            background: none;
            border: none;
            color: #888;
            cursor: pointer;
            font-size: 16px;
            padding: 0 2px;
            line-height: 1;
        }
        .${EVICTION_TOAST_CLASS} .ev-close:hover { color: #ddd; }
        .${EVICTION_TOAST_CLASS} .ev-body {
            font-size: 12px;
            color: #bbb;
        }
        .${EVICTION_TOAST_CLASS} .ev-file {
            display: flex;
            justify-content: space-between;
            gap: 8px;
            padding: 2px 0;
            border-bottom: 1px solid #2a2a2a;
            word-break: break-all;
        }
        .${EVICTION_TOAST_CLASS} .ev-file:last-child { border-bottom: none; }
        .${EVICTION_TOAST_CLASS} .ev-size { white-space: nowrap; color: #888; }
        .${EVICTION_TOAST_CLASS} .ev-total {
            margin-top: 8px;
            font-size: 11px;
            color: #888;
            text-align: right;
        }
    `;
    document.head.appendChild(style);
}

function loadEvictionPos() {
    try {
        const raw = localStorage.getItem(EVICTION_POS_KEY);
        if (!raw) return null;
        const pos = JSON.parse(raw);
        if (typeof pos.bottom !== "number" || typeof pos.right !== "number") return null;
        return pos;
    } catch { return null; }
}

function saveEvictionPos(bottom, right) {
    try { localStorage.setItem(EVICTION_POS_KEY, JSON.stringify({ bottom, right })); } catch { /**/ }
}

function positionEvictionToast(el) {
    const pos = loadEvictionPos() || { bottom: 20, right: 20 };
    el.style.bottom = `${pos.bottom}px`;
    el.style.right = `${pos.right}px`;
    el.style.top = "auto";
    el.style.left = "auto";
}

function setupEvictionDrag(toast) {
    const header = toast.querySelector(".ev-header");
    let dragging = false, startX, startY, startBottom, startRight;
    header.addEventListener("mousedown", (e) => {
        if (e.target.closest(".ev-close")) return;
        dragging = true;
        const rect = toast.getBoundingClientRect();
        startX = e.clientX; startY = e.clientY;
        startBottom = window.innerHeight - rect.bottom;
        startRight = window.innerWidth - rect.right;
        header.classList.add("dragging");
        e.preventDefault();
    });
    document.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        toast.style.bottom = `${startBottom - (e.clientY - startY)}px`;
        toast.style.right = `${startRight - (e.clientX - startX)}px`;
        toast.style.top = "auto"; toast.style.left = "auto";
    });
    document.addEventListener("mouseup", () => {
        if (!dragging) return;
        dragging = false;
        header.classList.remove("dragging");
        const rect = toast.getBoundingClientRect();
        saveEvictionPos(window.innerHeight - rect.bottom, window.innerWidth - rect.right);
    });
}

function dismissEviction() {
    if (!activeEvictionToast) return;
    activeEvictionToast.classList.add("hiding");
    const el = activeEvictionToast;
    setTimeout(() => { el.remove(); if (activeEvictionToast === el) activeEvictionToast = null; }, 300);
}

function formatBytes(bytes) {
    if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
    if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
    return `${(bytes / 1024).toFixed(0)} KB`;
}

function showEvictionToast(evicted, totalFreedBytes) {
    injectEvictionStyles();
    if (activeEvictionToast) activeEvictionToast.remove();

    const el = document.createElement("div");
    el.className = EVICTION_TOAST_CLASS;

    const fileRows = evicted.map(({ filename, size_bytes }) => `
        <div class="ev-file">
            <span>${filename}</span>
            <span class="ev-size">${formatBytes(size_bytes)}</span>
        </div>
    `).join("");

    el.innerHTML = `
        <div class="ev-header">
            <span class="ev-title">TensorRT \u2014 Disk Cleanup</span>
            <button class="ev-close" title="Dismiss">\u00d7</button>
        </div>
        <div class="ev-body">${fileRows}</div>
        <div class="ev-total">Freed ${formatBytes(totalFreedBytes)} total</div>
    `;
    el.querySelector(".ev-close").addEventListener("click", dismissEviction);
    document.body.appendChild(el);
    setupEvictionDrag(el);
    positionEvictionToast(el);
    activeEvictionToast = el;
}

// --- Progress toast ---

app.registerExtension({
    name: "ComfyUI.TensorRT.AutoLoaderToast",

    async setup() {
        api.addEventListener("trt_disk_eviction", ({ detail }) => {
            showEvictionToast(detail.evicted, detail.total_freed_bytes);
        });

        api.addEventListener("trt_build_progress", ({ detail }) => {
            console.log("[TRT build_progress] raw detail:", JSON.stringify(detail));
            if (!activeToast) {
                console.log("[TRT build_progress] no activeToast, ignoring");
                return;
            }
            const { phase_name, step, step_total, phase_idx, phase_count } = detail;
            const detailEl = activeToast.querySelector(".toast-build-detail");
            if (!detailEl) return;
            detailEl.style.display = "block";
            const phaseLabel = phase_count > 0 ? `[${phase_idx}/${phase_count}] ` : "";
            detailEl.textContent = `${phaseLabel}${phase_name}: ${step}/${step_total}`;
            // Update bar based on build step progress when no ETA
            if (currentEta == null && step_total > 0) {
                const pct = Math.min(95, (step / step_total) * 100);
                const fill = activeToast.querySelector(".toast-bar-fill");
                if (fill) fill.style.width = `${pct}%`;
            }
        });

        api.addEventListener("trt_auto_progress", ({ detail }) => {
            console.log("[TRT auto_progress] raw detail:", JSON.stringify(detail));
            const { phase, elapsed_s, eta_s } = detail;
            const info = PHASES[phase];
            if (!info) {
                console.warn("[TRT auto_progress] unknown phase:", phase, "— not in PHASES map");
                return;
            }

            const toast = getToast();
            console.log("[TRT auto_progress] phase=%s elapsed_s=%s eta_s=%s typeof_eta=%s", phase, elapsed_s, eta_s, typeof eta_s);

            // Cancel pending dismiss while still in progress
            if (dismissTimer && phase !== "done" && phase !== "cached") {
                clearTimeout(dismissTimer);
                dismissTimer = null;
            }

            const color = PHASE_COLORS[phase] || "#76B900";
            toast.querySelector(".toast-title").style.color = color;
            toast.querySelector(".toast-stage").textContent = info.text;
            toast.querySelector(".toast-bar-fill").style.background = color;

            // Hide build detail when leaving build phase
            const buildDetail = toast.querySelector(".toast-build-detail");
            if (buildDetail && phase !== "building") buildDetail.style.display = "none";

            if (phase === "done" || phase === "cached") {
                console.log("[TRT auto_progress] terminal phase=%s elapsed_s=%s detail=%s", phase, elapsed_s, JSON.stringify(detail));
                stopTick();
                currentEta = null;
                toast.querySelector(".toast-bar-fill").style.width = "100%";

                // Hide timer row, show summary instead
                const timerEl = toast.querySelector(".toast-timer");
                if (timerEl) timerEl.style.display = "none";

                // Build completion summary
                const { model_name, model_type: mtype, source, profile } = detail;
                const origEta = detail.eta_s;

                if (phase === "cached") {
                    toast.querySelector(".toast-stage").textContent = "Engine cached \u2014 skipped";
                } else if (model_name) {
                    // Compose summary
                    const sourceLabel = source || "unknown";
                    let line1 = `Loaded TRT ${mtype || ""}:${model_name}`;
                    if (profile) line1 += ` ${profile}`;

                    let line2 = `Took ${formatDuration(elapsed_s)}`;
                    let varianceHtml = "";
                    if (origEta != null && elapsed_s != null) {
                        const diff = elapsed_s - origEta;
                        const absDiff = Math.abs(diff);
                        const pctVar = origEta > 0 ? Math.round((diff / origEta) * 100) : 0;
                        const sign = diff > 0 ? "+" : "-";

                        // Color: bright green unless >10% AND >5s slower
                        let cls = "eta-faster";
                        if (diff > 0) {
                            cls = (diff > 5 && pctVar > 10) ? "eta-slower-warn" : "eta-slower-ok";
                        }
                        const label = diff > 0 ? "slower" : "faster";
                        varianceHtml = `, <span class="${cls}">${formatDuration(absDiff)} ${label} (${sign}${Math.abs(pctVar)}%)</span> vs ETA ${formatDuration(origEta)}`;
                    }

                    toast.querySelector(".toast-stage").innerHTML =
                        `<div style="font-size:12px;margin-bottom:4px">${line1}</div>` +
                        `<div class="toast-summary">${line2}${varianceHtml} via ${sourceLabel}</div>`;
                }

                toast.classList.add("done", "done-summary");
                if (!persistent) {
                    dismissTimer = setTimeout(dismiss, AUTO_DISMISS_MS);
                }
            } else if (elapsed_s != null) {
                // New operation starting — set ETA and start timer
                currentEta = eta_s != null ? eta_s : null;
                console.log("[TRT auto_progress] new operation phase=%s currentEta=%s", phase, currentEta);
                startTick();
                if (currentEta == null) {
                    toast.querySelector(".toast-bar-fill").style.width = `${info.pct}%`;
                } else {
                    toast.querySelector(".toast-bar-fill").style.width = "0%";
                }
            } else {
                // Phase text update only — timer continues running
                console.log("[TRT auto_progress] phase text update: %s (timer continues, currentEta=%s)", phase, currentEta);
            }
        });
    },
});
