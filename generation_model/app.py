import streamlit as st
import torch
import os
import io
import traceback
from ase.io import read, write
from ase.optimize import BFGS

# OFFICIAL CHGNet IMPORT
from chgnet.model.dynamics import CHGNetCalculator 

# LOCAL IMPORTS
from model_utils import GPT, GPTConfig
from CIFTokenizer import CIFTokenizer 
from mcts import MCTSSampler, MCTSEvaluator, PUCTSelector, ContextSensitiveTreeBuilder
from scorer import HeuristicPhysicalScorer

# --- Official CHGNet Relaxation & Analysis Function ---
def analyze_and_relax(cif_string, device):
    """
    Analyzes the initial energy/forces and relaxes the structure 
    to find real-world stability.
    """
    try:
        # ROBUST TYPE CHECK: Ensure we have a proper string
        if isinstance(cif_string, (bytes, bytearray)):
            cif_string = cif_string.decode('utf-8')
        elif not isinstance(cif_string, str):
            cif_string = str(cif_string)

        # Strip null bytes or stray characters that can corrupt the CIF
        cif_string = cif_string.replace('\x00', '').strip()

        chgnet_device = "cpu" if device == "mps" else device

        # Use BytesIO — ASE's CIF reader is more reliable with bytes than StringIO
        cif_bytes = cif_string.encode('utf-8')
        struct = read(io.BytesIO(cif_bytes), format='cif')

        # Extract formula for UI feedback
        formula = struct.get_chemical_formula()

        # Attach CHGNet Calculator
        calc = CHGNetCalculator(use_device=chgnet_device)
        struct.calc = calc

        # 1. Capture Raw (Initial) State
        raw_energy = struct.get_potential_energy()
        raw_forces = struct.get_forces()
        max_force_initial = torch.tensor(raw_forces).abs().max().item()

        # 2. Perform Relaxation (Structure Optimization)
        dyn = BFGS(struct, logfile=None)
        dyn.run(fmax=0.1, steps=50)

        # 3. Capture Final State
        final_energy = struct.get_potential_energy()
        energy_per_atom = final_energy / len(struct)

        # Write optimized structure to BytesIO, then decode for Streamlit
        out_buf = io.BytesIO()
        write(out_buf, struct, format='cif')
        optimized_cif = out_buf.getvalue().decode('utf-8')

        return {
            "cif": optimized_cif,
            "formula": formula,
            "raw_energy": raw_energy,
            "final_energy": final_energy,
            "energy_per_atom": energy_per_atom,
            "max_force": max_force_initial,
            "success": True
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- Backend Loading ---
@st.cache_resource
def load_backend():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt_path = "/Users/anvitha/Desktop/battery/generation_model/ckpt.pt"

    if not os.path.exists(ckpt_path):
        return None, None, device, f"Checkpoint not found at: {ckpt_path}"

    try:
        tokenizer = CIFTokenizer()
        config = GPTConfig()
        model = GPT(config)

        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint['model']

        for k in list(state_dict.keys()):
            if k.startswith('_orig_mod.'):
                state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        return model, tokenizer, device, "Success"
    except Exception as e:
        return None, None, device, str(e)


# --- UI Setup ---
st.set_page_config(page_title="Crystal-Gen Pro: Feasibility Mode", layout="wide")
st.title("Battery Crystal Discovery: Real-Life Feasibility")
st.markdown("This AI-powered laboratory proposes and evaluates **new chemical arrangements** for battery energy storage.")

with st.spinner("Initializing AI & Physics Engines..."):
    model, tokenizer, device, status_msg = load_backend()

if model is None:
    st.error(f"Initialization Failed: {status_msg}")
    st.stop()

# Sidebar
st.sidebar.header("System Status")
st.sidebar.success(f"Hardware: {device.upper()}")

st.sidebar.divider()
st.sidebar.header("Search Settings")
num_sims = st.sidebar.slider("MCTS Simulations", 10, 1000, 300)
width = st.sidebar.slider("Branching Width (Top-K)", 5, 100, 30)
temp = st.sidebar.slider("Creativity (Temperature)", 0.1, 1.5, 0.8)

st.sidebar.subheader("Target Material Profile")
target_rho = st.sidebar.slider("Target Density (g/cm³)", 1.0, 10.0, 3.5)
c_puct = st.sidebar.number_input("Exploration Constant (C-PUCT)", value=1.4)

# Main Interface
st.info("Discovery Engine: Currently in 'Open Discovery' mode starting from 'data_'.")

if st.button("Start Discovery & Analysis", type="primary"):
    start_prompt = "data_"

    try:
        # MCTS Setup
        external_scorer = HeuristicPhysicalScorer(target_density=target_rho)
        evaluator = MCTSEvaluator(scorer=external_scorer, tokenizer=tokenizer)
        selector = PUCTSelector(cpuct=c_puct)
        tree_builder = ContextSensitiveTreeBuilder(tokenizer=tokenizer)

        sampler = MCTSSampler(
            model=model, config=model.config, width=width, max_depth=512,
            eval_function=evaluator, node_selector=selector,
            tokenizer=tokenizer, temperature=temp, device=device,
            tree_builder=tree_builder
        )

        # 1. MCTS Search
        with st.status("AI exploring chemical landscape...", expanded=True) as status:
            sampler.search(start=start_prompt, num_simulations=num_sims)
            status.update(label="Discovery Phase Complete!", state="complete", expanded=False)

        # 2. Results Handling
        best_data = sampler.get_best_sequence()
        if best_data:
            best_seq, best_score = best_data

            # Decode and explicitly ensure string type
            cif_output = tokenizer.decode(best_seq)
            if isinstance(cif_output, (bytes, bytearray)):
                cif_output = cif_output.decode('utf-8')
            elif not isinstance(cif_output, str):
                cif_output = str(cif_output)

            if len(cif_output) < 200:
                st.error("Generated structure is incomplete. Please increase simulations.")
            else:
                # 3. CHGNet Analysis
                with st.spinner("Running Thermodynamic Stability Analysis..."):
                    results = analyze_and_relax(cif_output, device)

                if results["success"]:
                    st.success(f"Candidate Identified: **{results['formula']}**")

                    # --- FEASIBILITY DASHBOARD (2 metrics only) ---
                    m1, m2 = st.columns(2)

                    m1.metric(
                        label="Formation Energy",
                        value=f"{results['energy_per_atom']:.3f} eV/atom",
                        delta=f"{results['final_energy'] - results['raw_energy']:.3f} eV (Relaxed)",
                        delta_color="inverse"
                    )

                    m2.metric(
                        label="Structural Tension",
                        value=f"{results['max_force']:.3f} eV/Å",
                        help="Lower is better. Forces > 1.0 suggest an unstable lattice."
                    )

                    st.divider()

                    # --- SYNTHESIS PROBABILITY (full width) ---
                    st.subheader("Synthesis Probability")
                    energy = results['energy_per_atom']
                    if energy < -4.0:
                        st.success("**HIGH FEASIBILITY**: Energetically stable. This material is a strong candidate for laboratory synthesis.")
                    elif -4.0 <= energy < -1.0:
                        st.warning("**METASTABLE**: This structure may exist under specific pressure/temperature conditions but is less stable.")
                    else:
                        st.error("**UNSTABLE**: Energetically unfavorable. This structure is unlikely to form in a real-world environment.")

                    st.divider()

                    # --- ENERGY PROFILE ---
                    with st.expander("View Energy Profile"):
                        st.write(f"Pre-Optimization Energy:  {results['raw_energy']:.4f} eV")
                        st.write(f"Post-Optimization Energy: {results['final_energy']:.4f} eV")
                        st.caption("A large energy drop indicates the optimizer successfully fixed atomic overlaps.")

                    st.divider()

                    # --- FULL CIF FILE ---
                    st.subheader("Generated CIF Structure")
                    st.code(results['cif'], language="text")

                    st.download_button(
                        label="Download Optimized CIF",
                        data=results['cif'],
                        file_name=f"{results['formula']}_stable.cif",
                        mime="text/plain"
                    )

                else:
                    st.error(f"CHGNet Stability Error: {results['error']}")
        else:
            st.error("The search tree failed to converge on a valid crystal. Try increasing Width or Simulations.")

    except Exception:
        st.error("The Discovery Engine encountered a fatal error.")
        st.code(traceback.format_exc())