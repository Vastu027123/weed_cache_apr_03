import os
import yaml
from ultralytics import YOLO

def update_yaml_path(yaml_path, new_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    data['path'] = new_path
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def main():
    base_dir = "/root/weed_cache_apr_03/Downloads/Project_101/weed_0423"
    yaml_path = os.path.join(base_dir, "data.yaml")
    dataset_path = os.path.join(base_dir, "dataset")
    
    update_yaml_path(yaml_path, dataset_path)
    
    model = YOLO("yolo26m.pt") 
    
    model.train(
        data=yaml_path,
        epochs=200,         
        imgsz=640,          
        batch=64,           
        device=0,           
        workers=12,         
        optimizer='auto',   
        cache=True,         
        mosaic=1.0,         
        mixup=0.1,          
        degrees=5.0,        
        translate=0.05,     
        scale=0.2,          
        fliplr=0.5,         
        project="weed_detection_runs",
        name="yolo26m_train"
    )
    
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map:.4f}")

if __name__ == '__main__':
    main()

