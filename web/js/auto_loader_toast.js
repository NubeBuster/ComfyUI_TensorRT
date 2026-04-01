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
        .${TOAST_CLASS} .toast-close {
            background: none;
            border: none;
            color: #888;
            cursor: pointer;
            font-size: 16px;
            padding: 0 2px;
            line-height: 1;
        }
        .${TOAST_CLASS} .toast-close:hover {
            color: #ddd;
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
        .${TOAST_CLASS}.done .toast-title {
            color: #4CAF50;
        }
        .${TOAST_CLASS}.done .toast-bar-fill {
            background: #4CAF50;
        }
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
        if (e.target.closest(".toast-close")) return;
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
        const newRight = startRight - (e.clientX - startX);
        const newBottom = startBottom - (e.clientY - startY);
        toast.style.bottom = `${newBottom}px`;
        toast.style.right = `${newRight}px`;
        toast.style.top = "auto";
        toast.style.left = "auto";
    });

    document.addEventListener("mouseup", () => {
        if (!dragging) return;
        dragging = false;
        header.classList.remove("dragging");
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
            <button class="toast-close" title="Dismiss">\u00d7</button>
        </div>
        <div class="toast-stage">Preparing\u2026</div>
        <div class="toast-bar-track">
            <div class="toast-bar-fill" style="width: 0%"></div>
        </div>
    `;
    el.querySelector(".toast-close").addEventListener("click", dismiss);
    document.body.appendChild(el);
    setupDrag(el);
    positionToast(el);
    return el;
}

function getToast() {
    if (activeToast) {
        activeToast.classList.remove("hiding", "done");
        return activeToast;
    }
    activeToast = createToast();
    return activeToast;
}

function dismiss() {
    if (!activeToast) return;
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
    searching:  "#76B900",
    building:   "#4A9EFF",
    loading:    "#76B900",
    refitting:  "#76B900",
    done:       "#4CAF50",
    cached:     "#4CAF50",
};

const PHASES = {
    searching:  { text: "Searching for engine\u2026",              pct: 5  },
    building:   { text: "Building TRT engine (5\u201310 min)\u2026", pct: 10 },
    loading:    { text: "Loading engine\u2026",                     pct: 70 },
    refitting:  { text: "Refitting LoRA weights\u2026",             pct: 80 },
    done:       { text: "Done",                                     pct: 100 },
    cached:     { text: "Engine cached \u2014 skipped",             pct: 100 },
};

app.registerExtension({
    name: "ComfyUI.TensorRT.AutoLoaderToast",

    async setup() {
        api.addEventListener("trt_auto_progress", ({ detail }) => {
            const { phase } = detail;
            const info = PHASES[phase];
            if (!info) return;

            const toast = getToast();

            // Cancel pending dismiss while still in progress
            if (dismissTimer && phase !== "done" && phase !== "cached") {
                clearTimeout(dismissTimer);
                dismissTimer = null;
            }

            const color = PHASE_COLORS[phase] || "#76B900";
            toast.querySelector(".toast-title").style.color = color;
            toast.querySelector(".toast-stage").textContent = info.text;
            toast.querySelector(".toast-bar-fill").style.width = `${info.pct}%`;
            toast.querySelector(".toast-bar-fill").style.background = color;

            if (phase === "done" || phase === "cached") {
                toast.classList.add("done");
                dismissTimer = setTimeout(dismiss, AUTO_DISMISS_MS);
            }
        });
    },
});
