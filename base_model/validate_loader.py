import sys

sys.path.insert(0, "base_model")

import dataset
from dataset import ArrangementDataset

def main():
    ds = ArrangementDataset(data_dir="data")
    x = ds[0]

    print("OK dataset[0] retornou tipo:", type(x))

    if isinstance(x, (list, tuple)):
        print("len(item) =", len(x))
        for i, xi in enumerate(x):
            if hasattr(xi, "shape"):
                print(i, "shape:", xi.shape, "dtype:", getattr(xi, "dtype", None))
            else:
                print(i, "type:", type(xi))

if __name__ == "__main__":
    main()

