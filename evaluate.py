from ultralytics import YOLO


model_path = "best.pt"

model = YOLO(model_path)

# Evaluate the model on the validation set from data.yaml
metrics = model.val()

print("===== OVERALL METRICS =====")
print(f"mAP50       : {metrics.map50:.3f}")
print(f"mAP50-95    : {metrics.map:.3f}")
print(f"Precision   : {metrics.mp:.3f}")
print(f"Recall      : {metrics.mr:.3f}")

print("\n===== PER-CLASS METRICS =====")
print(f"{'Class':20s} {'Precision':10s} {'Recall':10s} {'AP50':10s} {'AP50-95':10s}")
print("-" * 65)
for i, class_name in enumerate(metrics.names):
    precision = metrics.p[i]
    recall = metrics.r[i]
    ap50 = metrics.ap50[i]
    ap = metrics.ap[i]
    print(
        f"{class_name:20s} {precision:<10.3f} {recall:<10.3f} {ap50:<10.3f} {ap:<10.3f}"
    )
