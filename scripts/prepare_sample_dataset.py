"""
Prepare Sample Night Traffic Dataset for YOLOv8 Training
Creates a structured dataset with images and YOLO-formatted label files (.txt)
covering night traffic classes: person, bicycle, car, motorcycle, bus, truck.
"""

import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "sample_night_traffic")

TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATASET_DIR, "train", "labels")
VAL_IMG_DIR = os.path.join(DATASET_DIR, "val", "images")
VAL_LBL_DIR = os.path.join(DATASET_DIR, "val", "labels")

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(TRAIN_LBL_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(VAL_LBL_DIR, exist_ok=True)

# Load base image
base_img_path = os.path.join(BASE_DIR, "data", "realistic_night_road.jpg")
assert os.path.exists(base_img_path), f"Base image not found at {base_img_path}"
base_img = cv2.imread(base_img_path)
H_orig, W_orig = base_img.shape[:2]

# Standardize to 640x640 for YOLO
TARGET_SIZE = (640, 640)

# Base ground-truth detections on realistic_night_road.jpg:
# Class mapping: 0: person, 1: bicycle, 2: car, 3: motorcycle, 4: bus, 5: truck
BASE_BOXES = [
    # (class_id, [x1, y1, x2, y2])
    (4, [159, 222, 471, 401]),     # bus
    (2, [451, 319, 629, 445]),     # car 1
    (2, [30, 301, 228, 447]),      # car 2
    (3, [982, 383, 1070, 514]),    # motorcycle
    (0, [970, 305, 1063, 472]),    # person (rider/pedestrian)
    (0, [1162, 300, 1206, 419]),   # person (sidewalk)
    (2, [684, 358, 827, 442]),     # car 3
]

def xyxy_to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / (2.0 * img_w)
    yc = (y1 + y2) / (2.0 * img_h)
    bw = (x2 - x1) / float(img_w)
    bh = (y2 - y1) / float(img_h)
    return round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)

def generate_variations():
    samples = []
    
    # 1. Base standard resize
    img_std = cv2.resize(base_img, TARGET_SIZE)
    boxes_std = []
    for cls_id, box in BASE_BOXES:
        x1 = box[0] * TARGET_SIZE[0] / W_orig
        y1 = box[1] * TARGET_SIZE[1] / H_orig
        x2 = box[2] * TARGET_SIZE[0] / W_orig
        y2 = box[3] * TARGET_SIZE[1] / H_orig
        boxes_std.append((cls_id, [x1, y1, x2, y2]))
    samples.append((img_std, boxes_std, "standard"))

    # 2. Darker low-light (gamma 0.5)
    lut_dark = np.array([((i / 255.0) ** 1.8) * 255 for i in range(256)]).astype("uint8")
    img_dark = cv2.LUT(img_std, lut_dark)
    samples.append((img_dark, boxes_std, "dark_gamma18"))

    # 3. Severe dark night (gamma 2.4)
    lut_dark2 = np.array([((i / 255.0) ** 2.4) * 255 for i in range(256)]).astype("uint8")
    img_dark2 = cv2.LUT(img_std, lut_dark2)
    samples.append((img_dark2, boxes_std, "severe_dark"))

    # 4. CLAHE Enhanced
    lab = cv2.cvtColor(img_std, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    samples.append((img_clahe, boxes_std, "clahe_enhanced"))

    # 5. Sensor noise + dark
    noise = np.random.normal(0, 15, img_dark.shape).astype(np.int16)
    img_noisy = np.clip(img_dark.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    samples.append((img_noisy, boxes_std, "dark_noisy"))

    # 6. Headlight Bloom / Warm tint
    img_warm = img_std.copy()
    img_warm[:, :, 0] = np.clip(img_warm[:, :, 0] * 0.85, 0, 255).astype(np.uint8) # reduce blue
    img_warm[:, :, 2] = np.clip(img_warm[:, :, 2] * 1.15, 0, 255).astype(np.uint8) # boost red
    samples.append((img_warm, boxes_std, "warm_headlights"))

    # 7. Horizontal Flip
    img_flipped = cv2.flip(img_std, 1)
    boxes_flipped = []
    for cls_id, (x1, y1, x2, y2) in boxes_std:
        nx1 = TARGET_SIZE[0] - x2
        nx2 = TARGET_SIZE[0] - x1
        boxes_flipped.append((cls_id, [nx1, y1, nx2, y2]))
    samples.append((img_flipped, boxes_flipped, "flipped"))

    # 8. Flipped + Dark
    img_flipped_dark = cv2.LUT(img_flipped, lut_dark)
    samples.append((img_flipped_dark, boxes_flipped, "flipped_dark"))

    # 9. Flipped + CLAHE
    lab_f = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2LAB)
    lf, af, bf = cv2.split(lab_f)
    clf = clahe.apply(lf)
    img_flipped_clahe = cv2.cvtColor(cv2.merge((clf, af, bf)), cv2.COLOR_LAB2BGR)
    samples.append((img_flipped_clahe, boxes_flipped, "flipped_clahe"))

    # 10. Left crop (focus on bus and oncoming car)
    crop_w = int(W_orig * 0.65)
    crop_img = base_img[:, :crop_w]
    crop_resized = cv2.resize(crop_img, TARGET_SIZE)
    boxes_crop = []
    for cls_id, box in BASE_BOXES:
        if box[0] < crop_w:
            cx1 = box[0] * TARGET_SIZE[0] / crop_w
            cy1 = box[1] * TARGET_SIZE[1] / H_orig
            cx2 = min(crop_w, box[2]) * TARGET_SIZE[0] / crop_w
            cy2 = box[3] * TARGET_SIZE[1] / H_orig
            if (cx2 - cx1) > 10:
                boxes_crop.append((cls_id, [cx1, cy1, cx2, cy2]))
    samples.append((crop_resized, boxes_crop, "left_crop_bus"))

    # 11. Right crop (focus on motorcycle and pedestrians)
    crop_start = int(W_orig * 0.45)
    crop_r_img = base_img[:, crop_start:]
    crop_r_w = W_orig - crop_start
    crop_r_resized = cv2.resize(crop_r_img, TARGET_SIZE)
    boxes_r_crop = []
    for cls_id, box in BASE_BOXES:
        if box[2] > crop_start:
            rx1 = max(0, box[0] - crop_start) * TARGET_SIZE[0] / crop_r_w
            ry1 = box[1] * TARGET_SIZE[1] / H_orig
            rx2 = (box[2] - crop_start) * TARGET_SIZE[0] / crop_r_w
            ry2 = box[3] * TARGET_SIZE[1] / H_orig
            if (rx2 - rx1) > 10:
                boxes_r_crop.append((cls_id, [rx1, ry1, rx2, ry2]))
    samples.append((crop_r_resized, boxes_r_crop, "right_crop_motorcycle"))

    # 12. Synthetic truck injection variation
    img_truck = img_std.copy()
    # draw distant truck silhouette in lane
    tx1, ty1, tx2, ty2 = 340, 240, 420, 310
    cv2.rectangle(img_truck, (tx1, ty1), (tx2, ty2), (30, 35, 45), -1)
    cv2.rectangle(img_truck, (tx1 + 5, ty1 + 5), (tx2 - 5, ty2 - 25), (40, 50, 60), -1)
    cv2.circle(img_truck, (tx1 + 15, ty2 - 5), 4, (120, 180, 255), -1)
    cv2.circle(img_truck, (tx2 - 15, ty2 - 5), 4, (120, 180, 255), -1)
    boxes_truck = list(boxes_std) + [(5, [tx1, ty1, tx2, ty2])] # 5: truck
    samples.append((img_truck, boxes_truck, "truck_injected"))

    return samples

def main():
    samples = generate_variations()
    print(f"Generated {len(samples)} distinct sample variations.")

    # 80% train, 20% val
    num_val = max(2, len(samples) // 5)
    num_train = len(samples) - num_val

    train_samples = samples[:num_train]
    val_samples = samples[num_train:]

    print(f"Assigning {len(train_samples)} images to train, {len(val_samples)} to val.")

    # Save train set
    for idx, (img, boxes, name) in enumerate(train_samples):
        img_name = f"night_train_{idx:02d}_{name}.jpg"
        lbl_name = f"night_train_{idx:02d}_{name}.txt"
        
        cv2.imwrite(os.path.join(TRAIN_IMG_DIR, img_name), img)
        with open(os.path.join(TRAIN_LBL_DIR, lbl_name), "w") as f:
            for cls_id, box in boxes:
                xc, yc, bw, bh = xyxy_to_yolo(box, TARGET_SIZE[0], TARGET_SIZE[1])
                f.write(f"{cls_id} {xc} {yc} {bw} {bh}\n")

    # Save val set
    for idx, (img, boxes, name) in enumerate(val_samples):
        img_name = f"night_val_{idx:02d}_{name}.jpg"
        lbl_name = f"night_val_{idx:02d}_{name}.txt"
        
        cv2.imwrite(os.path.join(VAL_IMG_DIR, img_name), img)
        with open(os.path.join(VAL_LBL_DIR, lbl_name), "w") as f:
            for cls_id, box in boxes:
                xc, yc, bw, bh = xyxy_to_yolo(box, TARGET_SIZE[0], TARGET_SIZE[1])
                f.write(f"{cls_id} {xc} {yc} {bw} {bh}\n")

    # Write data.yaml with absolute paths to avoid path resolution ambiguity
    data_yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    yaml_content = f"""# Sample Night Traffic Dataset Configuration for YOLOv8
path: {DATASET_DIR.replace(os.sep, '/')}
train: train/images
val: val/images

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: bus
  5: truck
"""
    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset successfully created at: {DATASET_DIR}")
    print(f"data.yaml configured at: {data_yaml_path}")

if __name__ == "__main__":
    main()

