import os
import pickle
import gzip
from tqdm import tqdm

cif_folder = 'battery_cifs'
output_bundle = 'battery_data_raw.pkl.gz'

bundled_data = []

print("Bundling CIF files...")
for filename in tqdm(os.listdir(cif_folder)):
    if filename.endswith(".cif"):
        file_path = os.path.join(cif_folder, filename)
        with open(file_path, 'r') as f:
            cif_content = f.read()
            # Use the filename as the ID
            bundled_data.append((filename, cif_content))

with gzip.open(output_bundle, 'wb') as f:
    pickle.dump(bundled_data, f)

print(f"Done! Created {output_bundle}")