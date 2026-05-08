"""
Patches MainTinySwinTransformerLungCancer_LOCAL.ipynb to:
1. Add class weights to CrossEntropyLoss (fixes Malignant bias)
2. Change epochs from 20 to 30
3. Update scheduler T_max to 30
"""
import json

NOTEBOOK_PATH = r"e:\Tiny swin transformer\MainTinySwinTransformerLungCancer_LOCAL.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

changes_made = []

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    
    src = "".join(cell["source"])
    
    # ── Cell 6: Model init ─────────────────────────────────────────────────
    # Replace criterion + scheduler
    if "criterion = nn.CrossEntropyLoss(label_smoothing=0.1)" in src and "T_max=10" in src:
        new_lines = []
        for line in cell["source"]:
            if "criterion = nn.CrossEntropyLoss(label_smoothing=0.1)" in line:
                # Insert class-weight lines before criterion
                new_lines += [
                    "# Class weights to handle heavy imbalance: Benign~120, Malignant~1190, Normal~55\n",
                    "# weight = total / (n_classes * class_count)\n",
                    "_total_s = 120 + 1190 + 55\n",
                    "class_weights = torch.tensor([\n",
                    "    _total_s / (3 * 120),   # Benign cases  (index 0)\n",
                    "    _total_s / (3 * 1190),  # Malignant cases (index 1)\n",
                    "    _total_s / (3 * 55),    # Normal cases  (index 2)\n",
                    "    1.0,                    # Unused 4th output neuron\n",
                    "], dtype=torch.float).to(device)\n",
                    "criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)\n",
                ]
            elif "T_max=10" in line:
                new_lines.append(line.replace("T_max=10", "T_max=30"))
            else:
                new_lines.append(line)
        cell["source"] = new_lines
        changes_made.append(f"Cell {i}: Added class weights + T_max=30")

    # ── Cell 7: train_model call ───────────────────────────────────────────
    if "train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=20)" in src:
        cell["source"] = [
            line.replace("num_epochs=20", "num_epochs=30")
            for line in cell["source"]
        ]
        changes_made.append(f"Cell {i}: num_epochs 20→30 in train_model call")

    # ── Cell 9: run_tracked_experiment ────────────────────────────────────
    if "criterion_run = nn.CrossEntropyLoss(label_smoothing=0.1)" in src:
        new_lines = []
        for line in cell["source"]:
            if "criterion_run = nn.CrossEntropyLoss(label_smoothing=0.1)" in line:
                new_lines += [
                    "        # Class weights: Benign~120, Malignant~1190, Normal~55\n",
                    "        _t = 120 + 1190 + 55\n",
                    "        _cw = torch.tensor([_t/(3*120), _t/(3*1190), _t/(3*55), 1.0],\n",
                    "                           dtype=torch.float).to(device)\n",
                    "        criterion_run = nn.CrossEntropyLoss(weight=_cw, label_smoothing=0.1)\n",
                ]
            else:
                new_lines.append(line)
        cell["source"] = new_lines
        changes_made.append(f"Cell {i}: Added class weights to criterion_run")

    if "run_tracked_experiment(epochs=20," in src:
        cell["source"] = [
            line.replace("epochs=20,", "epochs=30,")
            for line in cell["source"]
        ]
        changes_made.append(f"Cell {i}: epochs 20→30 in run_tracked_experiment call")

# Also clear all outputs so the notebook runs fresh
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Patch complete. Changes made:")
for c in changes_made:
    print(" ", c)

if not changes_made:
    print("  WARNING: No changes were made - check the target strings!")
