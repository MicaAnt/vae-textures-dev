import glob, os, pickle

npz_dir = "data/POP09-PIANOROLL-4-bin-quantization"
paths = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))

assert len(paths) > 0, f"Nenhum .npz encontrado em {npz_dir}"
with open("data/ind.pkl", "wb") as f:
    pickle.dump(paths, f)

print("OK ind.pkl criado:", len(paths), "arquivos")
print("Exemplo:", paths[0])


