import os
import glob
from collections import defaultdict
import yaml

def get_metrics(dataset_path, yaml_path):
    # Load class names from data.yaml if it exists
    class_names = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            try:
                data = yaml.safe_load(f)
                if 'names' in data:
                    if isinstance(data['names'], list):
                        class_names = {str(i): name for i, name in enumerate(data['names'])}
                    elif isinstance(data['names'], dict):
                        class_names = {str(k): v for k, v in data['names'].items()}
            except Exception as e:
                print(f"Warning: Could not parse {yaml_path}. Error: {e}")

    labels_dir = os.path.join(dataset_path, 'labels')
    images_dir = os.path.join(dataset_path, 'images')
    
    # Count total image files (searching for common extensions)
    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(images_dir, '**', ext), recursive=True))
    total_images = len(image_files)
    
    # Process label files
    label_files = glob.glob(os.path.join(labels_dir, '**', '*.txt'), recursive=True)
    total_annotated_images = len(label_files)
    
    total_instances = 0
    instances_per_class = defaultdict(int)
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # YOLO format: class_id x_center y_center width height
                class_id = line.split()[0]
                instances_per_class[class_id] += 1
                total_instances += 1
                
    print("="*40)
    print("DATASET METRICS")
    print("="*40)
    print(f"Total Images (in images folder): {total_images}")
    print(f"Total Annotated Images (.txt files): {total_annotated_images}")
    print(f"Total Instances (objects): {total_instances}")
    print("-" * 40)
    print("Instances per class:")
    
    for cls_id, count in sorted(instances_per_class.items(), key=lambda x: int(x[0])):
        cls_name = class_names.get(cls_id, f"Class {cls_id}")
        print(f"  {cls_name} (ID: {cls_id}): {count}")
    print("="*40)

if __name__ == "__main__":
    base_dir = "/root/weed_cache_apr_03/Downloads/Project_101/weed_0423"
    dataset_path = os.path.join(base_dir, "dataset")
    yaml_path = os.path.join(base_dir, "data.yaml")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found.")
    else:
        # We need PyYAML installed to parse the data.yaml
        try:
            import yaml
            get_metrics(dataset_path, yaml_path)
        except ImportError:
            print("PyYAML is missing. Please run 'pip install pyyaml' to map class names.")
            # Run anyway without pyyaml mapping correctly
            get_metrics(dataset_path, yaml_path)
