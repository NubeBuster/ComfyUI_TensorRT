/**
 * TensorRT Loader Auto — Widget visibility toggle
 *
 * Three-level toggle:
 * 1. on_missing -> shows/hides all build widgets (static_shapes, context_len, dimensions)
 * 2. static_shapes -> shows static OR dynamic dimension widgets
 * 3. disk_management -> shows/hides max_disk_usage_gb
 */

import { app } from "/scripts/app.js";

const STATIC_WIDGETS = ["height", "width", "batch_size"];

const DYNAMIC_WIDGETS = [
    "min_height", "opt_height", "max_height",
    "min_width", "opt_width", "max_width",
    "min_batch", "opt_batch", "max_batch",
];

// Widgets visible when on_missing is "build" (plus static/dynamic subsets)
const BUILD_ONLY_WIDGETS = ["static_shapes", "context_len", "disk_management"];

const DISK_WIDGETS = ["threshold_gb"];

function setWidgetHidden(widget, hidden) {
    if (widget._originalType === undefined) {
        widget._originalType = widget.type;
        widget._originalComputeSize = widget.computeSize;
    }
    if (hidden) {
        widget.type = "hidden";
        widget.hidden = true;
        widget.computeSize = () => [0, -4];
    } else {
        widget.type = widget._originalType;
        widget.hidden = false;
        widget.computeSize = widget._originalComputeSize;
    }
}

function updateWidgetVisibility(node) {
    const buildWidget = node.widgets?.find(w => w.name === "on_missing");
    const staticWidget = node.widgets?.find(w => w.name === "static_shapes");
    const diskWidget = node.widgets?.find(w => w.name === "disk_management");

    const buildEnabled = (buildWidget?.value ?? "build") === "build";
    const isStatic = (staticWidget?.value ?? "static") === "static";
    const diskEnabled = (diskWidget?.value ?? "disabled") !== "disabled";

    for (const widget of node.widgets || []) {
        if (BUILD_ONLY_WIDGETS.includes(widget.name)) {
            setWidgetHidden(widget, !buildEnabled);
        } else if (STATIC_WIDGETS.includes(widget.name)) {
            setWidgetHidden(widget, !buildEnabled || !isStatic);
        } else if (DYNAMIC_WIDGETS.includes(widget.name)) {
            setWidgetHidden(widget, !buildEnabled || isStatic);
        } else if (DISK_WIDGETS.includes(widget.name)) {
            setWidgetHidden(widget, !buildEnabled || !diskEnabled);
        }
    }

    node.setSize(node.computeSize());
    if (node.graph?.canvas) {
        node.graph.canvas.setDirty(true, true);
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "ComfyUI.TensorRT.AutoLoader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "TensorRTLoaderAuto") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // Attach callbacks to toggle widgets
            for (const toggleName of ["on_missing", "static_shapes", "disk_management"]) {
                const toggle = this.widgets?.find(w => w.name === toggleName);
                if (toggle) {
                    const originalCallback = toggle.callback;
                    toggle.callback = (value) => {
                        if (originalCallback) originalCallback.call(this, value);
                        updateWidgetVisibility(this);
                    };
                }
            }

            setTimeout(() => updateWidgetVisibility(this), 0);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (o) {
            const result = onConfigure?.apply(this, arguments);
            setTimeout(() => updateWidgetVisibility(this), 0);
            return result;
        };
    }
});
