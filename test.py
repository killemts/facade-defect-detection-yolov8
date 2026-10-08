from ultralytics import YOLO

# 1. Modelimizi yüklüyoruz
model = YOLO("best.pt")

# 2. Tek bir resim yerine doğrudan klasörün kendisini hedef gösteriyoruz
# Böylece test/images içindeki TÜM görselleri sırayla tahmin edecek
test_klasoru_yolu = "test/images"

# conf=0.25 yaparak modelin biraz daha hassas (esnek) arama yapmasını sağlıyoruz
results = model.predict(source=test_klasoru_yolu, save=True, conf=0.25)

print("\n=========================================")
print("🎯 Toplu test bitti!")
print("Tüm sonuçları runs/detect/predict klasörünün içinde görebilirsin.")
print("=========================================\n")