import os
import glob
import cv2
import random
import yaml
import albumentations as A
import uuid
from collections import defaultdict

def load_yolo_label(label_path):
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                bboxes.append([x_center, y_center, width, height, class_id])
    return bboxes

def save_yolo_label(label_path, bboxes):
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            f.write(f"{int(bbox[4])} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

def get_instance_counts(labels_dir):
    label_files = glob.glob(os.path.join(labels_dir, '**', '*.txt'), recursive=True)
    counts = defaultdict(int)
    image_to_classes = defaultdict(set)
    
    for label_file in label_files:
        bboxes = load_yolo_label(label_file)
        for bbox in bboxes:
            class_id = int(bbox[4])
            counts[class_id] += 1
            image_to_classes[label_file].add(class_id)
            
    return counts, image_to_classes, label_files

def augment_dataset(dataset_path, target_instances=5000):
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')
    
    # Define augmentations based on user requests
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=15, p=1.0)
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))

    current_counts, image_to_classes, all_labels = get_instance_counts(labels_dir)
    print("Initial counts:", dict(current_counts))
    
    target_classes = []
    for cls in [0, 1, 2]:
        if current_counts[cls] < target_instances:
            target_classes.append(cls)
            
    if not target_classes:
        print("All classes already have the target number of instances.")
        return

    # Find which image files correspond to which label files
    label_to_image = {}
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(file)[0]
                # Try to map to label by matching directory structure or just filename
                # Simplified matching assuming flat structure inside train/val/test
                rel_path = os.path.relpath(root, images_dir)
                label_path = os.path.join(labels_dir, rel_path, base_name + '.txt')
                if os.path.exists(label_path):
                    label_to_image[label_path] = os.path.join(root, file)

    for target_class in target_classes:
        print(f"\nAugmenting class {target_class} from {current_counts[target_class]} to {target_instances}")
        
        # Get all labels that contain this target class
        candidate_labels = [label for label, classes in image_to_classes.items() if target_class in classes]
        candidate_labels = [label for label in candidate_labels if label in label_to_image]
        
        if not candidate_labels:
            print(f"Warning: No images found containing class {target_class}. Skipping.")
            continue
            
        while current_counts[target_class] < target_instances:
            # Pick a random candidate image
            label_file = random.choice(candidate_labels)
            image_file = label_to_image[label_file]
            
            image = cv2.imread(image_file)
            if image is None:
                continue
                
            bboxes = load_yolo_label(label_file)
            if not bboxes:
                continue
                
            class_labels = [int(box[4]) for box in bboxes]
            bboxes_xywh = [box[:4] for box in bboxes]
            
            # Apply augmentation
            try:
                transformed = transform(image=image, bboxes=bboxes_xywh, class_labels=class_labels)
                aug_image = transformed['image']
                aug_bboxes_xywh = transformed['bboxes']
                aug_class_labels = transformed['class_labels']
            except Exception as e:
                # Sometimes bounding boxes can be invalid or fall outside, skip
                continue
                
            if not aug_bboxes_xywh:
                continue
                
            # Reconstruct boxes
            aug_bboxes = []
            for bbox_xywh, cls_id in zip(aug_bboxes_xywh, aug_class_labels):
                aug_bboxes.append(list(bbox_xywh) + [cls_id])
                
            # Check if target class is still in the image after augmentation (it might be cut off)
            if target_class not in aug_class_labels:
                continue
                
            # Save new image and label
            uid = uuid.uuid4().hex[:8]
            base_dir_img = os.path.dirname(image_file)
            base_dir_lbl = os.path.dirname(label_file)
            orig_name = os.path.splitext(os.path.basename(image_file))[0]
            
            new_img_name = f"{orig_name}_aug_{uid}.jpg"
            new_lbl_name = f"{orig_name}_aug_{uid}.txt"
            
            new_img_path = os.path.join(base_dir_img, new_img_name)
            new_lbl_path = os.path.join(base_dir_lbl, new_lbl_name)
            
            cv2.imwrite(new_img_path, aug_image)
            save_yolo_label(new_lbl_path, aug_bboxes)
            
            # Update counts
            for cls_id in aug_class_labels:
                current_counts[cls_id] += 1
                
        print(f"Finished class {target_class}. New count: {current_counts[target_class]}")

    print("\nFinal counts:", dict(current_counts))

if __name__ == "__main__":
    base_dir = "/root/weed_cache_apr_03/Downloads/Project_101/weed_0423"
    dataset_path = os.path.join(base_dir, "dataset")
    augment_dataset(dataset_path, 5000)
