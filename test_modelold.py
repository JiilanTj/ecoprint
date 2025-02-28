import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model yang telah dilatih
model = load_model('model.keras')  # Ganti dengan path model Anda jika diperlukan

# Fungsi untuk mempersiapkan gambar
def prepare_image(image_path):
    # Muat gambar
    img = cv2.imread(image_path)
    # Ubah ukuran gambar
    img = cv2.resize(img, (224, 224))
    # Normalisasi gambar
    img = img / 255.0
    # Ubah bentuk gambar menjadi (1, 224, 224, 3)
    img = np.reshape(img, (1, 224, 224, 3))
    return img

# Ganti dengan path gambar yang ingin diuji
image_path = './kamboja1.jpeg'  # Ganti dengan path gambar Anda
prepared_image = prepare_image(image_path)

# Melakukan prediksi
predictions = model.predict(prepared_image)
predicted_class = np.argmax(predictions, axis=1)

# Menampilkan hasil prediksi
print(f'Predicted class index: {predicted_class[0]}')

# Menampilkan probabilitas untuk setiap kelas
print(f'Predictions: {predictions}') 