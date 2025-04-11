import numpy as np

fpath = "/home/davy/Ensta/PIE/Terrain/Terrain_Detection/src/data/annotations/pts_dict_1047001_New.npy"

raw = np.load(fpath, allow_pickle=True)

print("Tipo:", type(raw))
print("Shape:", getattr(raw, "shape", "sem shape"))
print("Conteúdo:")
print(raw)

# Tenta extrair manualmente o dicionário
if isinstance(raw, np.ndarray):
    for i, item in enumerate(raw):
        print(f"Item {i}: {type(item)}")
        if isinstance(item, dict):
            print("Chaves do dicionário:", item.keys())
