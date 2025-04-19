# ====================
# better matching but haven't tested training
# ====================

import json
from pathlib import Path
import time
import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


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

    # 1. Ego kart
    qa_pairs.append({"image_file": image_file, "question": "What kart is the ego car?", "answer": ego["kart_name"]})

    # 2. Count
    qa_pairs.append({"image_file": image_file, "question": "How many karts are there in the scenario?", "answer": str(len(karts))})

    # 3. Track
    qa_pairs.append({"image_file": image_file, "question": "What track is this?", "answer": track_name})

    # 4. Relative positions
    for k in karts:
        if k["instance_id"] == ego["instance_id"]:
            continue

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
            # "answer": "in front of" if dy > 0 else "behind"
            
        })

    # 5. Counting
    left = sum(1 for k in karts if k["center"][0] < ego["center"][0] and k != ego)
    right = sum(1 for k in karts if k["center"][0] > ego["center"][0] and k != ego)
    front = sum(1 for k in karts if k["center"][1] < ego["center"][1] and k != ego)
    behind = sum(1 for k in karts if k["center"][1] > ego["center"][1] and k != ego)

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

    print(f"🔍 Found {len(info_files)} info files in data/{split}/")

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


# ====================
# Trainable code but poor quality
# ====================

# import json
# from pathlib import Path
# import time

# import fire
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image, ImageDraw

# OBJECT_TYPES = {
#     1: "Kart",
#     2: "Track Boundary",
#     3: "Track Element",
#     4: "Special Element 1",
#     5: "Special Element 2",
#     6: "Special Element 3",
# }

# COLORS = {
#     1: (0, 255, 0),
#     2: (255, 0, 0),
#     3: (0, 0, 255),
#     4: (255, 255, 0),
#     5: (255, 0, 255),
#     6: (0, 255, 255),
# }

# ORIGINAL_WIDTH = 600
# ORIGINAL_HEIGHT = 400


# def extract_frame_info(image_path: str) -> tuple[int, int]:
#     filename = Path(image_path).name
#     parts = filename.split("_")
#     if len(parts) >= 2:
#         frame_id = int(parts[0], 16)
#         view_index = int(parts[1])
#         return frame_id, view_index
#     return 0, 0


# def extract_kart_objects(info_path: str, view_index: int, img_width=150, img_height=100) -> list:
#     with open(info_path) as f:
#         info = json.load(f)

#     detections = info["detections"][view_index]
#     karts = []
#     image_center = (img_width / 2, img_height / 2)

#     for d in detections:
#         class_id, track_id, x1, y1, x2, y2 = d
#         if int(class_id) != 1:
#             continue

#         center_x = ((x1 + x2) / 2) * img_width / ORIGINAL_WIDTH
#         center_y = ((y1 + y2) / 2) * img_height / ORIGINAL_HEIGHT

#         karts.append({
#             "instance_id": int(track_id),
#             "kart_name": f"Kart {int(track_id)}",
#             "center": (center_x, center_y)
#         })

#     if karts:
#         def dist(k): return (k["center"][0] - image_center[0]) ** 2 + (k["center"][1] - image_center[1]) ** 2
#         ego = min(karts, key=dist)
#         for k in karts:
#             k["is_center_kart"] = (k is ego)

#     return karts


# def extract_track_info(info_path: str) -> str:
#     with open(info_path) as f:
#         info = json.load(f)
#     return info.get("track_name", "Unknown Track")


# def generate_qa_pairs(info_path: str, view_index: int, img_width=150, img_height=100) -> list:
#     karts = extract_kart_objects(info_path, view_index, img_width, img_height)
#     track_name = extract_track_info(info_path)
#     qa_pairs = []

#     if not karts:
#         return qa_pairs

#     ego = next(k for k in karts if k.get("is_center_kart"))
#     # image_file = f"{Path(info_path).stem.replace('_info', '')}_{view_index:02d}_im.jpg"
#     image_file = f"{Path(info_path).parent.name}/{Path(info_path).stem.replace('_info', '')}_{view_index:02d}_im.jpg"


#     # 1. Ego kart
#     qa_pairs.append({"image_file": image_file, "question": "What kart is the ego car?", "answer": ego["kart_name"]})

#     # 2. Count
#     qa_pairs.append({"image_file": image_file, "question": "How many karts are there in the scenario?", "answer": str(len(karts))})

#     # 3. Track
#     qa_pairs.append({"image_file": image_file, "question": "What track is this?", "answer": track_name})

#     # 4. Relative positions
#     for k in karts:
#         if k["instance_id"] == ego["instance_id"]:
#             continue

#         dx = k["center"][0] - ego["center"][0]
#         dy = ego["center"][1] - k["center"][1]

#         qa_pairs.append({
#             "image_file": image_file,
#             "question": f"Is {k['kart_name']} to the left or right of the ego car?",
#             "answer": "right" if dx > 0 else "left"
#         })

#         qa_pairs.append({
#             "image_file": image_file,
#             "question": f"Is {k['kart_name']} in front of or behind the ego car?",
#             "answer": "in front of" if dy > 0 else "behind"
#         })

#     # 5. Counting
#     left = sum(1 for k in karts if k["center"][0] < ego["center"][0] and k != ego)
#     right = sum(1 for k in karts if k["center"][0] > ego["center"][0] and k != ego)
#     front = sum(1 for k in karts if k["center"][1] < ego["center"][1] and k != ego)
#     behind = sum(1 for k in karts if k["center"][1] > ego["center"][1] and k != ego)

#     for label, count in [("left", left), ("right", right), ("in front of", front), ("behind", behind)]:
#         qa_pairs.append({
#             "image_file": image_file,
#             "question": f"How many karts are {label} the ego car?",
#             "answer": str(count)
#         })

#     return qa_pairs


# def check_qa_pairs(info_file: str, view_index: int):
#     """
#     Check QA pairs for a specific info file and view index.

#     Args:
#         info_file: Path to the info.json file
#         view_index: Index of the view to analyze
#     """
#     # Find corresponding image file
#     info_path = Path(info_file)
#     base_name = info_path.stem.replace("_info", "")
#     image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

#     # Visualize detections
#     annotated_image = draw_detections(str(image_file), info_file)

#     # Display the image
#     plt.figure(figsize=(12, 8))
#     plt.imshow(annotated_image)
#     plt.axis("off")
#     plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
#     plt.show()

#     # Generate QA pairs
#     qa_pairs = generate_qa_pairs(info_file, view_index)

#     # Print QA pairs
#     print("\nQuestion-Answer Pairs:")
#     print("-" * 50)
#     for qa in qa_pairs:
#         print(f"Q: {qa['question']}")
#         print(f"A: {qa['answer']}")
#         print("-" * 50)


# def generate_dataset(split: str):
#     data_dir = Path("data") / split
#     info_files = list(sorted(data_dir.glob("*_info.json")))

#     all_written = 0
#     start_time = time.time()

#     print(f"🔍 Found {len(info_files)} info files in data/{split}/")

#     for idx, info_file in enumerate(info_files, 1):
#         all_qas = []

#         for view_index in range(5):
#             try:
#                 qas = generate_qa_pairs(str(info_file), view_index)
#                 all_qas.extend(qas)
#             except Exception as e:
#                 print(f"⚠️ Skipping {info_file.name} view {view_index}: {e}")
#                 continue

#         if all_qas:
#             out_path = data_dir / f"{info_file.stem.replace('_info', '')}_qa_pairs.json"
#             with open(out_path, "w") as f:
#                 json.dump(all_qas, f, indent=2)

#             print(f"✅ [{idx}/{len(info_files)}] Wrote {len(all_qas)} QAs to {out_path.name}")

#         # ETA logging
#         elapsed = time.time() - start_time
#         per_file = elapsed / idx
#         eta = per_file * (len(info_files) - idx)
#         print(f"⏱ Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s ({eta/60:.1f} min)")

#         all_written += len(all_qas)

#     print(f"\n🎉 Done. Total QA pairs written: {all_written}")


# def main():
#     fire.Fire({
#         "check": check_qa_pairs,
#         "generate_dataset": generate_dataset,
#     })


# if __name__ == "__main__":
#     main()
