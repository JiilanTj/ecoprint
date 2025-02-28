import cv2
import numpy as np
from tensorflow.keras.models import load_model
from colorama import Fore, Style, init
from datetime import datetime

# Inisialisasi colorama
init(autoreset=True)

# Load model yang telah dilatih
model = load_model('model1.keras')  # Ganti dengan path model Anda jika diperlukan

# Daftar nama kategori sesuai dengan urutan indeks
categories = [
    'Bambujapan',  # Indeks 0
    'Binahong',    # Indeks 1
    'Bodhi',       # Indeks 2
    'Jarak Merah', # Indeks 3
    'Jati',        # Indeks 4
    'Kamboja',     # Indeks 5
    'Kayu Afrika', # Indeks 6
    'Lanang',      # Indeks 7
    'Palem Jamrud',# Indeks 8
    'Tulak'        # Indeks 9
]

# Daftar daun yang cocok untuk ecoprint
ecoprint_compatible = ['Jati', 'Kayu Afrika', 'Bodhi', 'Lanang', 'Jarak Merah']
# Daftar daun yang tidak cocok untuk ecoprint
ecoprint_incompatible = ['Kamboja', 'Tulak', 'Bambujapan', 'Binahong', 'Palem Jamrud']

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
image_path = './mawar.jpeg'  # Ganti dengan path gambar Anda
prepared_image = prepare_image(image_path)

# Melakukan prediksi
predictions = model.predict(prepared_image)
predicted_class_index = np.argmax(predictions, axis=1)[0]
predicted_class_name = categories[predicted_class_index]

# Menentukan ambang batas probabilitas
threshold = 0.5  # Anda bisa menyesuaikan ambang batas ini

# Validasi hasil prediksi
if predictions[0][predicted_class_index] < threshold:
    print(Fore.RED + "Gambar tidak terdeteksi sebagai daun yang valid.")
else:
    # Menampilkan hasil prediksi
    print(Fore.GREEN + "\n=== Hasil Prediksi ===")
    print(Fore.CYAN + f'Predicted class: {predicted_class_name} (Index: {predicted_class_index})')

    # Mengurutkan probabilitas
    sorted_indices = np.argsort(predictions[0])[::-1]  # Mengurutkan dari yang tertinggi ke terendah

    # Menampilkan probabilitas untuk setiap kelas dalam urutan
    print(Fore.YELLOW + "\nProbabilities sorted from highest to lowest:")
    for i in sorted_indices:
        prob = predictions[0][i]
        print(Fore.WHITE + f'Probability of {categories[i]}: {prob:.4f}')

    # Cek apakah daun cocok untuk ecoprint
    if predicted_class_name in ecoprint_compatible:
        print(Fore.GREEN + f'\n{predicted_class_name} dapat digunakan untuk ecoprint.')
    elif predicted_class_name in ecoprint_incompatible:
        print(Fore.RED + f'\n{predicted_class_name} tidak dapat digunakan untuk ecoprint.')
    else:
        print(Fore.YELLOW + f'\n{predicted_class_name} tidak teridentifikasi dalam kategori ecoprint.') 