import os
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter

# --- SECURE CONFIGURATION ---
API_KEY = os.getenv("MP_API_KEY")
if not API_KEY:
    raise ValueError("Set MP_API_KEY environment variable.")

SAVE_DIR = "full_battery_dataset"
IONS = ["Li", "Na", "Mg", "K"]
METALS = ["Fe", "Mn", "Co", "Ni", "V", "Cr", "Ti", "Cu"]
STABILITY_THRESHOLD = 0.1  # eV/atom

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def download_all_battery_cifs():
    unique_materials = {}

    with MPRester(API_KEY) as mpr:
        for ion in IONS:
            print(f"\nFetching ALL stable {ion}-based oxide candidates...")
            
            # Search for materials with Ion, Oxygen, and a Redox Metal
            # We don't limit the count here
            docs = mpr.materials.summary.search(
                elements=[ion, "O"],
                energy_above_hull=(0, STABILITY_THRESHOLD),
                num_elements=(3, 5), # Ternary to Pentenary
                fields=["material_id", "structure", "formula_pretty"]
            )
            
            # Filtering for redox metals in the results
            for doc in docs:
                if any(m in doc.formula_pretty for m in METALS):
                    unique_materials[doc.material_id] = doc

        total = len(unique_materials)
        print(f"\n--- TOTAL UNIQUE BATTERY CANDIDATES: {total} ---")

        for count, (mpid, doc) in enumerate(unique_materials.items(), 1):
            filename = f"{doc.formula_pretty}_{mpid}.cif"
            filepath = os.path.join(SAVE_DIR, filename)
            
            # Only download if we don't already have it
            if not os.path.exists(filepath):
                try:
                    CifWriter(doc.structure).write_file(filepath)
                    if count % 100 == 0:
                        print(f"Progress: {count}/{total} files saved...")
                except Exception:
                    continue
            else:
                if count % 500 == 0:
                    print(f"Skipped {count} (already exists)...")

    print(f"\nDone! All {total} CIFs are in '{SAVE_DIR}'")

if __name__ == "__main__":
    download_all_battery_cifs()