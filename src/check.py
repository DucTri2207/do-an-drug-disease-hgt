from pathlib import Path
import torch
import pickle

def inspect_file(path_str):
    path = Path(path_str)
    print(f"\n========== {path.name} ==========")
    print("suffix:", path.suffix if path.suffix else "(no extension)")

    # thử đọc text
    try:
        text = path.read_text(encoding="utf-8")
        print("Looks like TEXT")
        print(text[:1000])
        return
    except Exception:
        pass

    # thử torch.load
    try:
        obj = torch.load(path, map_location="cpu")
        print("Loaded by torch")
        print("type:", type(obj))
        if isinstance(obj, dict):
            print("keys:", list(obj.keys()))
        else:
            print(obj)
        return
    except Exception:
        pass

    # thử pickle
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print("Loaded by pickle")
        print("type:", type(obj))
        print(obj)
        return
    except Exception:
        pass

    # in bytes đầu file
    with open(path, "rb") as f:
        head = f.read(64)
    print("Unknown binary file")
    print("first 64 bytes:", head)

inspect_file("baseline_inference")
inspect_file("hgt_inference_smoke.pt")