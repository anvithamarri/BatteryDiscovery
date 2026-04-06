import os
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter

# --- SECURE CONFIGURATION ---
API_KEY = os.getenv("MP_API_KEY")

if not API_KEY:
    raise ValueError("MP_API_KEY not found! Set it with: export MP_API_KEY='your_key'")

SAVE_DIR = "battery_cifs"
# We define our ions. We will search for each one individually to get the full list.
BATTERY_IONS = ["Li", "Na", "Mg", "K"]
STABILITY_THRESHOLD = 0.1 

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def download_battery_data():
    all_docs = []
    
    with MPRester(API_KEY) as mpr:
        for ion in BATTERY_IONS:
            print(f"Searching for {ion}-based materials...")
            # We search for materials containing 'ion' and at least one other element (e.g., Li-*)
            # energy_above_hull=(min, max)
            docs = mpr.materials.summary.search(
                elements=[ion], 
                energy_above_hull=(0, STABILITY_THRESHOLD),
                fields=["material_id", "structure", "formula_pretty"]
            )
            all_docs.extend(docs)
            print(f"Found {len(docs)} for {ion}.")

        # Remove duplicates (some materials might contain two of our target ions)
        unique_docs = {doc.material_id: doc for doc in all_docs}.values()
        print(f"Total unique materials to download: {len(unique_docs)}")

        count = 0
        for doc in unique_docs:
            # Format filename to be clean
            formula = doc.formula_pretty
            mpid = str(doc.material_id)
            filename = f"{formula}_{mpid}.cif".replace("/", "_")
            filepath = os.path.join(SAVE_DIR, filename)
            
            try:
                CifWriter(doc.structure).write_file(filepath)
                count += 1
                if count % 100 == 0:
                    print(f"Progress: {count}/{len(unique_docs)} downloaded...")
            except Exception as e:
                pass # Skip errors for individual files

    print(f"\nDone! {count} files saved to '{SAVE_DIR}'")

if __name__ == "__main__":
    download_battery_data()