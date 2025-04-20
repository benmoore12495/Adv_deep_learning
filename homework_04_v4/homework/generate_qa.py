import json
from pathlib import Path
import time
import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400
# MIN_WIDTH = 15
# MIN_HEIGHT = 15
MIN_WIDTH = 5
MIN_HEIGHT = 5
MIN_AREA = 100  # adjust if needed

def extract_frame_info(image_path: str) -> tuple[int, int]:
    filename = Path(image_path).name
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0

def extract_kart_objects(info_path: str, view_index: int, id_to_name: dict[int, str], img_width=150, img_height=100) -> list:
    with open(info_path) as f:
        info = json.load(f)

    detections = info["detections"][view_index]
    karts = []
    image_center = (img_width / 2, img_height / 2)

    for d in detections:
        class_id, track_id, x1, y1, x2, y2 = d
        if int(class_id) != 1:
            continue

        if (x2 - x1) < MIN_WIDTH or (y2 - y1) < MIN_HEIGHT:
            continue

        center_x = ((x1 + x2) / 2) * img_width / ORIGINAL_WIDTH
        center_y = ((y1 + y2) / 2) * img_height / ORIGINAL_HEIGHT

        karts.append({
            "instance_id": int(track_id),
            "kart_name": id_to_name.get(int(track_id), f"Kart {int(track_id)}"),
            "center": (center_x, center_y)
        })

    if karts:
        def dist(k): return (k["center"][0] - image_center[0]) ** 2 + (k["center"][1] - image_center[1]) ** 2
        ego = min(karts, key=dist)
        for k in karts:
            k["is_center_kart"] = (k is ego)

    return karts

def extract_track_info(info_path: str) -> tuple[str, dict[int, str]]:
    with open(info_path) as f:
        info = json.load(f)
    track_name = info.get("track", "Unknown Track")
    id_to_name = {i: name for i, name in enumerate(info.get("karts", []))}
    return track_name, id_to_name

def generate_qa_pairs(info_path: str, view_index: int, img_width=150, img_height=100) -> list:
    track_name, id_to_name = extract_track_info(info_path)
    karts = extract_kart_objects(info_path, view_index, id_to_name, img_width, img_height)
    qa_pairs = []

    if not karts:
        return qa_pairs

    ego = next(k for k in karts if k.get("is_center_kart"))
    image_file = f"{Path(info_path).parent.name}/{Path(info_path).stem.replace('_info', '')}_{view_index:02d}_im.jpg"

    qa_pairs.append({
        "image_file": image_file,
        "question": "What kart is the ego car?",
        "answer": ego["kart_name"]
    })

    with open(info_path) as f:
        info = json.load(f)
    detections = info["detections"][view_index]
    # unique_kart_ids = {
    #     track_id
    #     for class_id, track_id, x1, y1, x2, y2 in detections
    #     if class_id == 1 and (x2 - x1) >= MIN_WIDTH and (y2 - y1) >= MIN_HEIGHT
    # }
    unique_kart_ids = {
        track_id
        for class_id, track_id, x1, y1, x2, y2 in detections
        if class_id == 1
        and (x2 - x1) >= MIN_WIDTH
        and (y2 - y1) >= MIN_HEIGHT
        and (x2 - x1) * (y2 - y1) >= MIN_AREA
    }
    qa_pairs.append({
        "image_file": image_file,
        "question": "How many karts are there in the scenario?",
        "answer": str(len(unique_kart_ids))
    })

    qa_pairs.append({
        "image_file": image_file,
        "question": "What track is this?",
        "answer": track_name
    })

    visible_karts = [k for k in karts if k["instance_id"] != ego["instance_id"]]
    for k in visible_karts:
        dx = k["center"][0] - ego["center"][0]
        dy = ego["center"][1] - k["center"][1]

        qa_pairs.append({
            "image_file": image_file,
            "question": f"Is {k['kart_name']} to the left or right of the ego car?",
            "answer": "right" if dx > 0 else "left"
        })

        qa_pairs.append({
            "image_file": image_file,
            "question": f"Is {k['kart_name']} in front of or behind the ego car?",
            "answer": "front" if dy > 0 else "back"
        })

    left = sum(1 for k in visible_karts if k["center"][0] < ego["center"][0])
    right = sum(1 for k in visible_karts if k["center"][0] > ego["center"][0])
    front = sum(1 for k in visible_karts if k["center"][1] < ego["center"][1])
    behind = sum(1 for k in visible_karts if k["center"][1] > ego["center"][1])

    for label, count in [("left", left), ("right", right), ("in front of", front), ("behind", behind)]:
        qa_pairs.append({
            "image_file": image_file,
            "question": f"How many karts are {label} the ego car?",
            "answer": str(count)
        })

    return qa_pairs

def generate_dataset(split: str):
    data_dir = Path("data") / split
    info_files = list(sorted(data_dir.glob("*_info.json")))

    all_written = 0
    start_time = time.time()

    print(f"\U0001F50D Found {len(info_files)} info files in data/{split}/")

    for idx, info_file in enumerate(info_files, 1):
        all_qas = []

        for view_index in range(5):
            try:
                qas = generate_qa_pairs(str(info_file), view_index)
                all_qas.extend(qas)
            except Exception as e:
                print(f"⚠️ Skipping {info_file.name} view {view_index}: {e}")
                continue

        if all_qas:
            out_path = data_dir / f"{info_file.stem.replace('_info', '')}_qa_pairs.json"
            with open(out_path, "w") as f:
                json.dump(all_qas, f, indent=2)

            print(f"✅ [{idx}/{len(info_files)}] Wrote {len(all_qas)} QAs to {out_path.name}")

        elapsed = time.time() - start_time
        per_file = elapsed / idx
        eta = per_file * (len(info_files) - idx)
        print(f"⏱ Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s ({eta/60:.1f} min)")

        all_written += len(all_qas)

    print(f"\n🎉 Done. Total QA pairs written: {all_written}")

def main():
    fire.Fire({"generate_dataset": generate_dataset})

if __name__ == "__main__":
    main()

