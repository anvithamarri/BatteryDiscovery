import streamlit as st
import torch
import io
import zipfile
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

# -- CONSTANTS -----------------------------------------------
REFRACTORY = {"W","Mo","Ta","Nb","Re","Hf","Zr","V","Cr","Ti"}
PRECIOUS   = {"Au","Ag","Pt","Pd","Ir","Ru","Rh"}
NON_METALS = {"H","He","C","N","O","F","Ne","P","S","Cl","Ar","Se","Br","Kr","I","Xe","At","Rn"}

# -- HELPERS -------------------------------------------------
def classify_intermetallic(symbols):
    if any(s in NON_METALS for s in symbols): return None
    if all(s in REFRACTORY  for s in symbols): return "Refractory Alloy"
    if any(s in PRECIOUS    for s in symbols): return "Precious Metal Intermetallic"
    return "Standard Intermetallic"

def analyze_structure(cif_string, device):
    try:
        struct = read(io.BytesIO(cif_string.encode()), format="cif")
        symbols = list(set(struct.get_chemical_symbols()))
        alloy_type = classify_intermetallic(symbols)

        calc = CHGNetCalculator(use_device="cpu" if device == "mps" else device)
        struct.calc = calc

        dist_matrix = struct.get_all_distances(mic=True)
        if np.min(dist_matrix[np.nonzero(dist_matrix)]) < 1.5:
            return None

        dyn = BFGS(ExpCellFilter(struct), logfile=None)
        dyn.run(fmax=0.02, steps=200)

        energy = struct.get_potential_energy() / len(struct)
        force  = np.abs(struct.get_forces()).max()

        buf = io.BytesIO()
        write(buf, struct, format="cif")

        return {
            "formula":          struct.get_chemical_formula(),
            "cif":              buf.getvalue().decode(),
            "energy_per_atom":  energy,
            "force":            force,
            "is_intermetallic": alloy_type is not None,
            "type":             alloy_type,
            "valid":            alloy_type is not None and energy < 0 and force <= 0.05
        }
    except Exception:
        return None

def make_zip(dataset):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i, r in enumerate(dataset):
            if r["valid"]:
                zf.writestr(f"{i:03d}_{r['formula']}.cif", r["cif"])
    return buf.getvalue()

@st.cache_resource
def load_backend():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok  = CIFTokenizer()
    ckpt = torch.load("ckpt.pt", map_location=device)
    m    = GPT(GPTConfig())
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}, strict=False)
    m.to(device).eval()
    return m, tok, device

# -- UI ------------------------------------------------------
st.set_page_config(page_title="Intermetallic Factory", layout="wide")
st.title("Intermetallic Dataset Factory")

if "dataset" not in st.session_state:
    st.session_state.dataset = []

model, tokenizer, device = load_backend()

with st.sidebar:
    st.header("Configuration")
    dataset_size = st.number_input("Dataset Size", min_value=1, max_value=5000, value=50)
    run          = st.button("Generate")

if run:
    st.session_state.dataset = []
    found, attempts = 0, 0
    bar    = st.progress(0)
    status = st.empty()
    log    = st.expander("Generation Log", expanded=True)

    selector     = PUCTSelector(cpuct=1.4)
    tree_builder = ContextSensitiveTreeBuilder(tokenizer=tokenizer)
    evaluator    = MCTSEvaluator(scorer=HeuristicPhysicalScorer(), tokenizer=tokenizer)

    while attempts < dataset_size:
        attempts += 1
        status.text(f"Attempt {attempts}/{dataset_size} — Found {found} valid crystals...")

        sampler = MCTSSampler(
            model=model, config=model.config, width=30, max_depth=512,
            eval_function=evaluator, node_selector=selector,
            tokenizer=tokenizer, temperature=0.8, device=device,
            tree_builder=tree_builder
        )
        sampler.search("data_", num_simulations=10)
        best = sampler.get_best_sequence()

        if not best:
            log.write(f"Attempt {attempts:03d} | Failed generation")
            continue

        tokens  = best[0] if isinstance(best, tuple) else best
        cif_raw = tokenizer.decode(tokens)

        if len(cif_raw) < 200:
            log.write(f"Attempt {attempts:03d} | Invalid CIF")
            continue

        res = analyze_structure(cif_raw, device)

        if res is None:
            log.write(f"Attempt {attempts:03d} | Relaxation failed")
            continue

        st.session_state.dataset.append(res)
        if res["valid"]:
            found += 1

        log.write(
            f"Attempt {attempts:03d} | {res['formula']:<12} | "
            f"{res['type'] or 'Non-metallic':<30} | "
            f"E: {res['energy_per_atom']:.3f} eV/at | "
            f"F: {res['force']:.3f} eV/A | "
            f"{'VALID' if res['valid'] else 'REJECT'}"
        )

        bar.progress(attempts / dataset_size)

    st.success(f"Done. {found} valid crystals found in {attempts} attempts.")

# -- RESULTS -------------------------------------------------
if st.session_state.dataset:
    valid = [r for r in st.session_state.dataset if r["valid"]]
    total = len(st.session_state.dataset)

    st.header("Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Valid Found",         f"{found if run else len(valid)}/{dataset_size}")
    c2.metric("Intermetallic Yield", f"{sum(r['is_intermetallic'] for r in st.session_state.dataset)/total*100:.1f}%")
    c3.metric("Valid Rate",          f"{len(valid)/total*100:.1f}%")
    c4.metric("Avg Energy",          f"{np.mean([r['energy_per_atom'] for r in valid]):.3f} eV/atom" if valid else "N/A")
    c5.metric("Avg Max Force",       f"{np.mean([r['force'] for r in valid]):.3f} eV/A" if valid else "N/A")

    if valid:
        st.subheader("Alloy Type Diversity")
        diversity = Counter(r["type"] for r in valid)
        cols = st.columns(len(diversity))
        for col, (atype, count) in zip(cols, diversity.items()):
            col.metric(atype, count)

    st.download_button("Download ZIP", make_zip(st.session_state.dataset),
                       "intermetallics.zip", "application/zip")

    st.subheader("Preview (Top 5 Valid)")
    for r in valid[:5]:
        with st.expander(f"{r['formula']} — {r['type']} | E: {r['energy_per_atom']:.3f} eV/at"):
            st.code(r["cif"], language="text")
