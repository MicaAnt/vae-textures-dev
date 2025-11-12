# createNPZBatches.py

import os

def criar_batches_npz(input_folder="npzFiles", output_folder="batchesNPZ", batch_size=5):
    os.makedirs(output_folder, exist_ok=True)

    arquivos = sorted([f for f in os.listdir(input_folder) if f.endswith(".npz")])
    track_ids = [os.path.splitext(f)[0] for f in arquivos]

    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i:i + batch_size]
        batch_filename = os.path.join(output_folder, f"batch_{i//batch_size:03d}.txt")
        with open(batch_filename, "w") as f:
            f.write("\n".join(batch))

    print(f"{(len(track_ids) + batch_size - 1) // batch_size} batches criados em {output_folder}")

if __name__ == "__main__":
    criar_batches_npz("npzFiles", "batchesNPZ", 10)
