import os
import subprocess
import argparse

BATCH_DIR = "./COMMUDataset/batchesNPZ/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Número do batch inicial")
    args = parser.parse_args()

    # Lista e ordena batches
    batches = sorted(f for f in os.listdir(BATCH_DIR) if f.startswith("batch_") and f.endswith(".txt"))

    # Filtra a partir do batch inicial
    batches = [b for b in batches if int(b.split("_")[1].split(".")[0]) >= args.start]

    for batch_file in batches:
        batch_num = batch_file.split("_")[1].split(".")[0]
        print(f"\n🚀 Rodando batch {batch_num}...")
        subprocess.run(["python", "NEWcalcLatentBatchLoos.py", "--batch", batch_num], check=True)

    print("\n✅ Todos os batches processados.")

if __name__ == "__main__":
    main()

