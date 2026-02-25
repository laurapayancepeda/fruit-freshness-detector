import os
import shutil
import random

dataset_path = "Fruit Freshness Dataset"
target_path = "Dataset"

classes = {
    "Apple/fresh": 0,
    "Apple/rotten": 1,
    "Banana/fresh": 2,
    "Banana/rotten": 3,
    "Strawberry/fresh": 4,
    "Strawberry/rotten": 5,
}


for split in ["train", "val"]:
    os.makedirs(f"{target_path}/images/{split}", exist_ok=True)
    os.makedirs(f"{target_path}/labels/{split}", exist_ok=True)

split_ratio = 0.8

for class_path, class_id in classes.items():
    full_path = os.path.join(dataset_path, *class_path.split("/"))
    if not os.path.exists(full_path):
        print(f"Warning: {full_path} does not exist!")
        continue

    images = os.listdir(full_path)
    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for split, split_images in zip(["train", "val"], [train_images, val_images]):
        for img in split_images:
            src_img = os.path.join(full_path, img)
            dst_img = os.path.join(target_path, "images", split, img)
            shutil.copy(src_img, dst_img)

            # YOLO label
            label_name = img.rsplit(".", 1)[0] + ".txt"
            label_path = os.path.join(target_path, "labels", split, label_name)
            with open(label_path, "w") as f:
                # Format: <class_id> <x_center> <y_center> <width> <height>
                # Full image = 0.5 0.5 1.0 1.0
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

# data.yaml
with open(os.path.join(target_path, "data.yaml"), "w") as f:
    f.write(f"""
path: {target_path}
train: images/train
val: images/val

names:
  0: fresh_apple
  1: rotten_apple
  2: fresh_banana
  3: rotten_banana
  4: fresh_strawberry
  5: rotten_strawberry
""")

print("YOLO dataset ready in folder:", target_path)
