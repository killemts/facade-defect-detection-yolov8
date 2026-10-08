import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# =========================================================
# SAYFA AYARLARI
# =========================================================
st.set_page_config(
    page_title="Yapısal Hasar Tespit Sistemi",
    page_icon="🏢",
    layout="wide"
)

# =========================================================
# MODEL YÜKLEME
# =========================================================
@st.cache_resource
def load_model():
    model_path = "runs/detect/catlak_tespit_modeli/weights/best.pt"

    if not os.path.exists(model_path):
        st.error(f"Model bulunamadı: {model_path}")
        st.stop()

    return YOLO(model_path)

model = load_model()

# =========================================================
# SOL PANEL
# =========================================================
st.sidebar.title("📘 Proje Bilgileri")

st.sidebar.info("""
### 🏗️ Proje Adı
Drone Görüntülerinden Dış Cephe Yüzey Kusurlarının
Belirlenmesine Yönelik YOLO Tabanlı Bir Tespit Sistemi

### 👩‍🏫 Danışman
Doç. Dr. Zeynep Hasırcı Tuğcu

### 👩‍💻 Proje Ekibi
- Merve Sarıoğlu
- Sinem Özalp
- Öykü Avkıran
""")

# =========================================================
# ANA BAŞLIK
# =========================================================
st.title("🏢 Yapısal Hasar Tespit ve Analiz Sistemi")
st.write(
    "Fotoğraf veya video yükleyerek yapısal çatlakların "
    "otomatik olarak tespit edilmesini sağlayabilirsiniz."
)

# =========================================================
# ANALİZ TÜRÜ
# =========================================================
analysis_type = st.radio(
    "Analiz Türünü Seçin:",
    ["Görsel Analizi", "Video Analizi"]
)

# =========================================================
# YÜKLEME ALANI
# =========================================================
if analysis_type == "Görsel Analizi":
    uploaded_file = st.file_uploader(
        "Görsel Yükleyin",
        type=["jpg", "jpeg", "png"]
    )
else:
    uploaded_file = st.file_uploader(
        "Video Yükleyin",
        type=["mp4", "avi", "mov"]
    )

# =========================================================
# ANALİZ FONKSİYONU
# =========================================================
def analyze_image(image_np):
    # Model tahmini
    result = model.predict(
        source=image_np,
        conf=0.25,      # Eşik değeri
        iou=0.45,
        verbose=False
    )[0]

    # Tespit edilen kutu sayısı
    crack_count = len(result.boxes)

    # Ortalama güven skoru
    if crack_count > 0:
        confidences = result.boxes.conf.cpu().numpy()
        avg_conf = float(np.mean(confidences) * 100)
    else:
        avg_conf = 0.0

    # Kutulu görsel
    annotated = result.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return annotated, crack_count, avg_conf

# =========================================================
# ANALİZ
# =========================================================
if uploaded_file is not None:

    # ---------------- GÖRSEL ANALİZİ ----------------
    if analysis_type == "Görsel Analizi":
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        if st.button("🔍 Analizi Başlat"):
            annotated, crack_count, avg_conf = analyze_image(image_np)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📸 Analiz Sonucu")
                st.image(
                    annotated,
                    caption="Tespit edilen çatlaklar kırmızı kutular ile gösterilmiştir.",
                    use_container_width=True
                )

            with col2:
                st.subheader("📊 Tespit Raporu")

                if crack_count > 0:
                    st.metric("Çatlak Var mı?", "Evet")
                    st.metric("Tespit Edilen Çatlak Sayısı", crack_count)
                    st.metric("Ortalama Güven (%)", f"{avg_conf:.2f}")

                    st.warning(
                        f"{crack_count} adet çatlak tespit edildi."
                    )
                else:
                    st.metric("Çatlak Var mı?", "Hayır")
                    st.metric("Tespit Edilen Çatlak Sayısı", 0)
                    st.metric("Ortalama Güven (%)", "0.00")

                    st.success(
                        "Herhangi bir yapısal çatlak tespit edilmedi."
                    )

    # ---------------- VIDEO ANALİZİ ----------------
    else:
        st.video(uploaded_file)

        if st.button("🎥 Videoyu Analiz Et"):
            # Geçici dosya oluştur
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            # Videodan ilk kareyi al
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated, crack_count, avg_conf = analyze_image(frame_rgb)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("🎬 Video Analiz Sonucu (İlk Kare)")
                    st.image(
                        annotated,
                        caption="İlk kare üzerindeki tespitler",
                        use_container_width=True
                    )

                with col2:
                    st.subheader("📊 Tespit Raporu")

                    if crack_count > 0:
                        st.metric("Çatlak Var mı?", "Evet")
                        st.metric("Tespit Edilen Çatlak Sayısı", crack_count)
                        st.metric("Ortalama Güven (%)", f"{avg_conf:.2f}")

                        st.warning(
                            f"{crack_count} adet çatlak tespit edildi."
                        )
                    else:
                        st.metric("Çatlak Var mı?", "Hayır")
                        st.metric("Tespit Edilen Çatlak Sayısı", 0)
                        st.metric("Ortalama Güven (%)", "0.00")

                        st.success(
                            "Videoda herhangi bir yapısal çatlak tespit edilmedi."
                        )

            else:
                st.error("Video okunamadı.")