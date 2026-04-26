import torch
import os
import io
import numpy as np
from collections import Counter

from ase.io import read, write
from ase.optimize import BFGS
from ase.filters import ExpCellFilter

from chgnet.model.dynamics import CHGNetCalculator

from model_utils import GPT, GPTConfig
from CIFTokenizer import CIFTokenizer
from mcts import MCTSSampler, MCTSEvaluator, PUCTSelector, ContextSensitiveTreeBuilder
from scorer import HeuristicPhysicalScorer

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CKPT_PATH = "ckpt.pt"
OUTPUT_DIR = "https://drive.google.com/drive/folders/1ywiXTB9retGp0Qr384R59hX8J1gQO3lH?usp=drive_link"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================
def load_model():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError("Checkpoint not found")

    tokenizer = CIFTokenizer()
    config = GPTConfig()
    model = GPT(config)

    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    state_dict = checkpoint["model"]

    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state[k[len("_orig_mod."):]] = v
        else:
            clean_state[k] = v

    model.load_state_dict(clean_state, strict=False)
    model.to(DEVICE).eval()

    return model, tokenizer

# ============================================================
# RELAXATION + FORMATION ENERGY
# ============================================================
def analyze_structure(cif_string, device):
    try:
        struct = read(io.BytesIO(cif_string.encode("utf-8")), format="cif")

        # Attach CHGNet
        calc = CHGNetCalculator(use_device=("cpu" if device == "mps" else device))
        struct.calc = calc

        # Relaxation
        ecf = ExpCellFilter(struct)
        dyn = BFGS(ecf, logfile=None)
        dyn.run(fmax=0.02, steps=200)

        final_energy = struct.get_potential_energy()
        forces = struct.get_forces()
        max_force = np.abs(forces).max()

        energy_per_atom = final_energy / len(struct)

        return struct, {
            "formula": struct.get_chemical_formula(),
            "energy_per_atom": energy_per_atom,
            "force": max_force
        }

    except Exception:
        return None, None

# ============================================================
# MAIN DISCOVERY LOOP
# ============================================================
def run_discovery(total_runs=100, num_sims=20):
    model, tokenizer = load_model()

    selector = PUCTSelector(cpuct=1.4)
    tree_builder = ContextSensitiveTreeBuilder(tokenizer=tokenizer)
    external_scorer = HeuristicPhysicalScorer(target_density=3.5)
    evaluator = MCTSEvaluator(scorer=external_scorer, tokenizer=tokenizer)

    valid_count = 0

    for i in range(1, total_runs + 1):
        sampler = MCTSSampler(
            model=model,
            config=model.config,
            width=30,
            max_depth=512,
            eval_function=evaluator,
            node_selector=selector,
            tokenizer=tokenizer,
            temperature=0.8,
            device=DEVICE,
            tree_builder=tree_builder
        )

        sampler.search("data_", num_simulations=num_sims)
        best = sampler.get_best_sequence()

        if not best:
            print(f"Run {i:03d} | Failed generation")
            continue

        tokens = best[0] if isinstance(best, tuple) else best
        cif = tokenizer.decode(tokens)

                struct, analysis = analyze_structure(cif, DEVICE)

        if struct is None or analysis is None:
            print(f"Run {i:03d} | Relaxation failed")
            continue

        # --- VALIDITY CHECK ---
        is_valid = (
            analysis["energy_per_atom"] < 0 and
            analysis["force"] <= 0.05
        )

        if is_valid:
            try:
                path = f"{OUTPUT_DIR}/run_{i}.cif"
                write(path, struct)
                valid_count += 1
                print(f"Run {i:03d} | VALID → saved {path}")
            except Exception as e:
                print(f"Run {i:03d} | VALID but failed to save: {e}")
        else:
            print(f"Run {i:03d} | REJECT → not saved")


    print(f"\n✅ Finished {total_runs} runs. {valid_count} valid CIFs saved in {OUTPUT_DIR}")

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    run_discovery(total_runs=20, num_sims=10)
