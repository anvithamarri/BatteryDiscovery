import torch
import os
import io
import numpy as np
from collections import Counter

from ase.io import read
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

REFRACTORY = {"W", "Mo", "Ta", "Nb", "Re", "Hf", "Zr", "V", "Cr", "Ti"}
PRECIOUS   = {"Au", "Ag", "Pt", "Pd", "Ir", "Ru", "Rh"}
NON_METALS = {
    "H", "He", "C", "N", "O", "F", "Ne", "P", "S",
    "Cl", "Ar", "Se", "Br", "Kr", "I", "Xe", "At", "Rn"
}

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
# INTERMETALLIC CHECK
# ============================================================
def classify_intermetallic(symbols):
    if any(s in NON_METALS for s in symbols):
        return None

    if all(s in REFRACTORY for s in symbols):
        return "Refractory Alloy"

    if any(s in PRECIOUS for s in symbols):
        return "Precious Metal Intermetallic"

    return "Standard Intermetallic"

# ============================================================
# RELAXATION + FORMATION ENERGY
# ============================================================
def analyze_structure(cif_string, device):
    try:
        struct = read(io.BytesIO(cif_string.encode("utf-8")), format="cif")

        # Element symbols
        symbols = list(set(struct.get_chemical_symbols()))

        alloy_type = classify_intermetallic(symbols)
        is_intermetallic = alloy_type is not None

        # Attach CHGNet
        calc = CHGNetCalculator(use_device=("cpu" if device == "mps" else device))
        struct.calc = calc

        # Geometry sanity
        dist_matrix = struct.get_all_distances(mic=True)
        min_dist = np.min(dist_matrix[np.nonzero(dist_matrix)])

        if min_dist < 1.5:
            return None

        # Relaxation
        ecf = ExpCellFilter(struct)
        dyn = BFGS(ecf, logfile=None)
        dyn.run(fmax=0.02, steps=200)

        final_energy = struct.get_potential_energy()
        forces = struct.get_forces()
        max_force = np.abs(forces).max()

        energy_per_atom = final_energy / len(struct)

        return {
            "formula": struct.get_chemical_formula(),
            "energy_per_atom": energy_per_atom,   # treated as formation energy proxy
            "force": max_force,
            "is_intermetallic": is_intermetallic,
            "type": alloy_type
        }

    except Exception:
        return None

# ============================================================
# MAIN DISCOVERY LOOP
# ============================================================
def run_discovery(total_runs=100, num_sims=20):

    model, tokenizer = load_model()

    selector = PUCTSelector(cpuct=1.4)
    tree_builder = ContextSensitiveTreeBuilder(tokenizer=tokenizer)

    external_scorer = HeuristicPhysicalScorer(target_density=3.5)
    evaluator = MCTSEvaluator(scorer=external_scorer, tokenizer=tokenizer)

    results = []
    intermetallic_count = 0

    print("=" * 80)
    print(f"Running {total_runs} generations")
    print("=" * 80)

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

        if len(cif) < 200:
            print(f"Run {i:03d} | Invalid CIF")
            continue

        analysis = analyze_structure(cif, DEVICE)

        if analysis is None:
            print(f"Run {i:03d} | Relaxation failed")
            continue

        if analysis["is_intermetallic"]:
            intermetallic_count += 1

        # --- VALIDATION USING FORMATION ENERGY ---
        is_stable = analysis["energy_per_atom"] < 0
        is_low_force = analysis["force"] <= 0.05

        is_valid = (
            analysis["is_intermetallic"] and
            is_stable and
            is_low_force
        )

        results.append(analysis)

        print(
            f"Run {i:03d} | {analysis['formula']:<12} | "
            f"Intermetallic: {analysis['is_intermetallic']} | "
            f"E_form: {analysis['energy_per_atom']:.3f} | "
            f"Force: {analysis['force']:.3f} | "
            f"{'VALID' if is_valid else 'REJECT'}"
        )

    # ============================================================
    # METRICS
    # ============================================================
    valid = [
        r for r in results
        if r["is_intermetallic"] and r["energy_per_atom"] < 0 and r["force"] <= 0.05
    ]

    success_rate = (len(valid) / total_runs) * 100
    alloy_yield = (intermetallic_count / total_runs) * 100

    avg_energy = np.mean([r["energy_per_atom"] for r in valid]) if valid else 0
    avg_force = np.mean([r["force"] for r in valid]) if valid else 0

    diversity = Counter([r["type"] for r in valid])

    print("\n" + "=" * 80)
    print("FINAL METRICS")
    print("=" * 80)
    print(f"Intermetallic Yield        : {alloy_yield:.2f}%")
    print(f"Valid Intermetallic Rate   : {success_rate:.2f}%")
    print(f"Avg Formation Energy       : {avg_energy:.3f} eV/atom")
    print(f"Avg Force                 : {avg_force:.3f} eV/Å")
    print(f"Diversity                 : {dict(diversity)}")
    print("=" * 80)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    run_discovery(total_runs=20, num_sims=10)