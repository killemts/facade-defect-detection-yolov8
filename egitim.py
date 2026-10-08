from ultralytics import YOLO

def train():
    # YOLOv8 nano modeli yükleniyor
    model = YOLO("yolov8n.pt")

    # Model eğitimi
    results = model.train(
        data="data.yaml",   
        epochs=100,         
        imgsz=640,          
        batch=8,            
        device=0,           
        workers=4,          
        patience=20,        
        project="runs",
        name="catlak_tespiti"
    )

    # Model doğrulama
    metrics = model.val()

    print("\n===== EĞİTİM SONUÇLARI =====")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print("============================")

# Windows'ta multiprocessing için bu blok ŞARTTIR:
if __name__ == '__main__':
    train()