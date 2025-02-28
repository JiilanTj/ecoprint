import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load model pre-trained MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Tambahkan layer baru
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)  # Ganti 10 dengan jumlah kategori daun Anda

# Buat model baru
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layer dari base model
for layer in base_model.layers:
    layer.trainable = False

# Augmentasi data
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load dataset
train_generator = train_datagen.flow_from_directory(
    './dataset',  # Ganti dengan path dataset Anda
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Menampilkan statistik sebelum pelatihan
num_categories = len(train_generator.class_indices)
categories = list(train_generator.class_indices.keys())
num_images = train_generator.samples  # Jumlah gambar yang ditemukan
num_augmented_images = (400 * num_categories)  # Total gambar setelah augmentasi
steps_per_epoch = (num_augmented_images // 32)  # Total gambar dibagi batch_size

print("Statistik Sebelum Pelatihan:")
print(f"Jumlah Kategori: {num_categories}")
print(f"Nama Kategori: {categories}")
print(f"Jumlah Gambar Ditemukan: {num_images}")
print(f"Jumlah Gambar Setelah Augmentasi: {num_augmented_images}")
print(f"Epoch: 15")
print(f"Steps per Epoch: {steps_per_epoch}")

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=15)  # Sesuaikan epochs sesuai kebutuhan

# Menyimpan model dalam format .keras
model.save('model.keras')