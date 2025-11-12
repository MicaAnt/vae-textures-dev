import pandas as pd
import os

csv_path = "./CommuVAEDataset.csv"
batch_size = 5000
batch_dir = "./batches"

# Garante que o diretório existe
os.makedirs(batch_dir, exist_ok=True)

df = pd.read_csv(csv_path, index_col=0)
track_ids = df["id"].tolist()
batches = [track_ids[i:i + batch_size] for i in range(0, len(track_ids), batch_size)]

# Salva cada batch como .txt (um id por linha)
for i, batch in enumerate(batches):
    with open(f"{batch_dir}/batch_{i:03}.txt", "w") as f:
        for tid in batch:
            f.write(tid + "\n")

print(f"{len(batches)} batches salvos em '{batch_dir}'")
