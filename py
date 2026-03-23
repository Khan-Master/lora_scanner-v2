import os
import csv
from collections import Counter
from safetensors import safe_open


def detect_by_keys(path):
    PATTERNS = {
        "Flux": [
            "double_blocks", "single_blocks",
            "lora_transformer_double", "lora_transformer_single",
        ],
        "SDXL": [
            "lora_te1_", "lora_te2_",
            "lora_unet_input_blocks", "lora_unet_middle_block",
            "add_embedding", "label_emb",
        ],
        "SD1.5": [
            "lora_te_text_model",
            "lora_unet_down_blocks", "lora_unet_up_blocks",
        ],
        "SD2.x": [
            "lora_unet_input_blocks_4",
        ],
    }
    try:
        with safe_open(path, framework="numpy") as f:
            keys_raw = list(f.keys())
        keys = [k.lower() for k in keys_raw]

        scores = {arch: 0 for arch in PATTERNS}
        for arch, patterns in PATTERNS.items():
            for p in patterns:
                if any(p in k for k in keys):
                    scores[arch] += 1

        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best

        with safe_open(path, framework="numpy") as f:
            for key in keys_raw[:20]:
                shape = list(f.get_tensor(key).shape)
                if not shape:
                    continue
                m = max(shape)
                if m >= 3072: return "Flux"
                if m >= 1280: return "SDXL"
                if m >= 768:  return "SD1.5"

    except Exception as e:
        print("    [tensor error] " + str(e))
    return "Unknown"


def detect_model(path, metadata, filename):
    base_ver = (
        metadata.get("ss_base_model_version") or
        metadata.get("modelspec.architecture") or
        metadata.get("architecture") or ""
    ).lower()

    if "flux" in base_ver:  base = "Flux"
    elif "sdxl" in base_ver: base = "SDXL"
    elif "sd_2" in base_ver or "2.1" in base_ver: base = "SD2.x"
    elif "sd_1" in base_ver or "1.5" in base_ver: base = "SD1.5"
    else:
        text = (str(metadata) + " " + filename).lower()
        if "flux"  in text: base = "Flux"
        elif "sdxl" in text or "xl_base" in text: base = "SDXL"
        elif "sd_2" in text or "sd2" in text: base = "SD2.x"
        elif "sd_1" in text or "sd15" in text or "1.5" in text: base = "SD1.5"
        else:
            base = detect_by_keys(path)

    if base == "Unknown":
        return "Unknown"

    text = (str(metadata) + " " + filename).lower()
    subtypes = [
        ("pony", "Pony"), ("illustrious", "Illustrious"),
        ("illustration", "Illustrious"),
        ("noobai", "NoobAI"), ("animagine", "Animagine"),
        ("qwen", "Qwen"),
        ("anime", "Anime"), ("anything", "Anime"),
        ("realistic", "Realistic"), ("photo", "Realistic"),
    ]
    for kw, label in subtypes:
        if kw in text:
            return base + " - " + label
    return base


def extract_triggers(metadata):
    keywords = ("trigger", "tag", "prompt", "activation", "keyword")
    found = []
    for k, v in metadata.items():
        if any(kw in k.lower() for kw in keywords):
            v = str(v).strip()
            if v:
                found.append(v)
    return " | ".join(found) if found else "None"


def scan_folder(folder):
    results = []
    files_found = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.endswith(".safetensors"):
                files_found.append((root, f))

    if not files_found:
        print("No .safetensors files found.")
        return results

    for root, file in files_found:
        path     = os.path.join(root, file)
        rel_path = os.path.relpath(path, folder)

        try:
            with safe_open(path, framework="numpy") as f:
                metadata = f.metadata() or {}
        except Exception as e:
            print("[!] Cannot open " + file + ": " + str(e))
            metadata = {}

        model    = detect_model(path, metadata, file)
        triggers = extract_triggers(metadata)
        icon     = "?" if model == "Unknown" else "v"

        print("[" + icon + "] " + rel_path.ljust(55) + " " + model)

        if model == "Unknown":
            print("     metadata: " + (str(dict(list(metadata.items())[:3])) if metadata else "(empty)"))
            try:
                with safe_open(path, framework="numpy") as f:
                    keys = list(f.keys())[:10]
                print("     tensor keys:")
                for k in keys:
                    print("       " + k)
            except Exception:
                pass

        results.append((rel_path, model, triggers))
    return results


def save_txt(results, folder):
    path = os.path.join(folder, "lora_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        for name, model, triggers in results:
            f.write(name + "\n")
            f.write("  Model:    " + model + "\n")
            f.write("  Triggers: " + triggers + "\n\n")
    return path


def save_csv(results, folder):
    path = os.path.join(folder, "lora_report.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["File", "Model", "Triggers"])
        w.writerows(results)
    return path


if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))

    print("=" * 65)
    print("  LoRA Scanner")
    print("  Folder: " + folder)
    print("=" * 65)

    results = scan_folder(folder)

    if results:
        counts = Counter(model for _, model, _ in results)
        print("\n" + "-" * 65)
        print("  Total: " + str(len(results)) + " files")
        for model, cnt in sorted(counts.items()):
            mark = " (!)" if model == "Unknown" else ""
            print("  " + model.ljust(40) + str(cnt) + mark)
        print("-" * 65)
        txt      = save_txt(results, folder)
        csv_path = save_csv(results, folder)
        print("\nSaved:\n  " + txt + "\n  " + csv_path)
    else:
        print("Nothing found.")

    input("\nPress Enter to exit...")
