"""
Runs all training cells from MainTinySwinTransformerLungCancer_LOCAL.ipynb
by extracting the code and executing it in a single Python session.
This includes the class-weight fix and 30 epochs.
"""
import json, sys, os

NOTEBOOK_PATH = r"e:\Tiny swin transformer\MainTinySwinTransformerLungCancer_LOCAL.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Collect all code cell sources
code_cells = []
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if src.strip():  # skip empty cells
            code_cells.append((i, src))

print(f"Found {len(code_cells)} code cells to run.")
print("="*60)

# Set working directory to notebook folder
os.chdir(r"e:\Tiny swin transformer")
sys.path.insert(0, r"e:\Tiny swin transformer")

# Run each cell
g = {}
for i, (cell_idx, src) in enumerate(code_cells):
    print(f"\n{'='*60}")
    print(f"Running Cell {cell_idx} ({i+1}/{len(code_cells)})")
    print(f"{'='*60}")
    print(src[:300] + ("..." if len(src) > 300 else ""))
    print()
    try:
        exec(compile(src, f"<cell_{cell_idx}>", "exec"), g)
        print(f"[OK] Cell {cell_idx} completed.")
    except Exception as e:
        print(f"[ERROR] Cell {cell_idx} failed: {e}")
        import traceback
        traceback.print_exc()
        # For DVC / MLflow / visualization cells, continue
        # For critical cells (data loading, model, training), stop
        if cell_idx <= 8:  # critical cells
            print("Critical cell failed, stopping.")
            sys.exit(1)
        else:
            print("Non-critical cell, continuing...")

print("\n" + "="*60)
print("All cells completed!")
