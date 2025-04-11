from pathlib import Path
import numpy as np

annotation_dir = Path("src/data/annotations")

for fpath in sorted(annotation_dir.glob("pts_dict_*.npy")):
    try:
        data = np.load(fpath, allow_pickle=True)

        if isinstance(data, dict) and "pts" in data and "ident" in data:
            print(f"[OK] {fpath.name} já está OK.")
            continue

        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 2:
            print(f"[FIXANDO] {fpath.name} — adicionando 'ident'")
            fixed = {
                "pts": data,
                "ident": np.arange(len(data))
            }
            np.save(fpath, fixed)
            continue

        if isinstance(data, np.ndarray):
            if data.shape == () and isinstance(data.item(), dict):
                fixed = data.item()
                np.save(fpath, fixed)
                print(f"[✔] {fpath.name} corrigido (scalar dict).")
                continue
            if len(data) == 1 and isinstance(data[0], dict):
                fixed = data[0]
                np.save(fpath, fixed)
                print(f"[✔] {fpath.name} corrigido (1-element array).")
                continue

        print(f"[⚠️] {fpath.name} não pôde ser corrigido. Formato desconhecido.")

    except Exception as e:
        print(f"[ERRO] {fpath.name} — {e}")
