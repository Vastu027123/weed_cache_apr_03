import os
import glob
import random

def reduce_class_instances(dataset_path, target_class_id=1, target_count=5500):
    labels_dir = os.path.join(dataset_path, 'labels')
    images_dir = os.path.join(dataset_path, 'images')
    
    # Get all label files
    all_label_files = glob.glob(os.path.join(labels_dir, '**', '*.txt'), recursive=True)
    
    # Count current instances of the target class
    total_class_instances = 0
    aug_label_files = []
    
    for label_file in all_label_files:
        is_aug = '_aug_' in os.path.basename(label_file)
        if is_aug:
            aug_label_files.append(label_file)
            
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) == target_class_id:
                    total_class_instances += 1
                    
    print(f"Current instances of class {target_class_id}: {total_class_instances}")
    
    if total_class_instances <= target_count:
        print(f"Total instances are already at or below {target_count}. No reduction needed.")
        return
        
    to_remove = total_class_instances - target_count
    print(f"Need to remove {to_remove} instances of class {target_class_id} from augmented data.")
    
    # Shuffle augmented files to randomly drop instances
    random.shuffle(aug_label_files)
    
    removed_count = 0
    empty_files_to_delete = []
    
    for label_file in aug_label_files:
        if removed_count >= to_remove:
            break
            
        with open(label_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        target_lines = [line for line in lines if int(line.split()[0]) == target_class_id]
        other_lines = [line for line in lines if int(line.split()[0]) != target_class_id]
        
        if not target_lines:
            continue
            
        # We can remove up to 'remaining_to_remove' instances from this file
        remaining_to_remove = to_remove - removed_count
        
        if len(target_lines) <= remaining_to_remove:
            # Drop all instances of this class in this file
            removed_count += len(target_lines)
            keep_target_lines = []
        else:
            # Drop only what's needed
            keep_target_lines = target_lines[remaining_to_remove:]
            removed_count += remaining_to_remove
            
        new_lines = other_lines + keep_target_lines
        
        if not new_lines:
            # If the file has no bounding boxes left, mark for deletion
            empty_files_to_delete.append(label_file)
        else:
            # Write back the kept bounding boxes
            with open(label_file, 'w') as f:
                for line in new_lines:
                    f.write(f"{line}\n")
                    
    print(f"Successfully removed {removed_count} instances.")
    
    # Clean up files that have no bounding boxes left (including associated images)
    deleted_images = 0
    for label_file in empty_files_to_delete:
        os.remove(label_file)
        
        # Try to find and delete the associated image
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        # the path in images_dir should mirror labels_dir or be flat
        rel_label_path = os.path.relpath(os.path.dirname(label_file), labels_dir)
        possible_image_bases = [
            os.path.join(images_dir, rel_label_path, base_name),
            os.path.join(images_dir, base_name)
        ]
        
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            for img_base in possible_image_bases:
                if os.path.exists(img_base + ext):
                    os.remove(img_base + ext)
                    deleted_images += 1
                    
    print(f"Deleted {len(empty_files_to_delete)} label files and {deleted_images} image files that became completely empty.")

if __name__ == "__main__":
    base_dir = "/root/weed_cache_apr_03/Downloads/Project_101/weed_0423"
    dataset_path = os.path.join(base_dir, "dataset")
    # RPW class ID is 1
    reduce_class_instances(dataset_path, target_class_id=1, target_count=5500)
