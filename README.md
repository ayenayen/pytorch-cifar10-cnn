# AI & Deep Learning Portfolio

Repository ini berisi kumpulan proyek-proyek dasar portofolio *Artificial Intelligence*, *Computer Vision*, dan *Deep Learning*.

---

## 📁 Proyek 1: Object Detection dengan YOLOv4 dan OpenCV
**File Utama:** `OpenCV.ipynb`

Proyek ini adalah implementasi sistem Visi Komputer (*Computer Vision*) untuk mendeteksi berbagai macam objek di dalam sebuah gambar menggunakan model *Deep Learning* YOLOv4 (*You Only Look Once*).

**Fitur & Teknologi:**
- **OpenCV (cv2):** Digunakan untuk membaca gambar, memanipulasi ukuran (blob), dan menggambar Bounding Box (kotak hijau) di atas objek.
- **YOLOv4 (Darknet):** Menggunakan bobot (*weights*) pre-trained yang sangat akurat untuk mengenali 80 jenis kelas objek umum (seperti mobil, manusia, anjing, sepeda, dll).
- **Matplotlib:** Digunakan untuk menampilkan hasil akhir gambar secara langsung di Jupyter Notebook.
- **Cara Kerja:** Gambar diterjemahkan menjadi *Blob*, dimasukkan ke dalam Neural Network (*Forward Pass*), dan hasil deteksi disaring menggunakan *Confidence Threshold* (hanya menampilkan objek dengan tingkat keyakinan di atas 50%).
- **Visualisasi & Analisis:** Program secara otomatis menggambar kotak hijau (Bounding Box), memberikan label teks bahasa Inggris berserta tingkat keyakinannya, dan menghitung total keseluruhan objek yang ditemukan di gambar.

---

## 📁 Proyek 2: CIFAR-10 Image Classification dengan PyTorch
**File Utama:** `pytorch.ipynb`

Proyek ini adalah implementasi dasar *Convolutional Neural Network* (CNN) dari nol menggunakan framework PyTorch untuk mengklasifikasikan gambar dari dataset CIFAR-10.

**Fitur & Teknologi:**
- Dataset CIFAR-10 (60.000 gambar berwarna ukuran 32x32 dalam 10 kelas).
- **Arsitektur:** 2 Layer Konvolusi (`Conv2d`), Layer MaxPooling (`MaxPool2d`), Aktivasi ReLU, dan 2 Layer Fully Connected (`Linear`).
- **Hasil Training:** Model dilatih selama 5 Epoch menggunakan *Optimizer Adam* dan mencapai **Akurasi ~71%** pada data *testing* menggunakan akselerasi GPU (CUDA).

## 📁 Proyek 3: Transfer Learning & Fine-Tuning dengan ResNet50 (Keras)
**File Utama:** `transferlearn.ipynb`

Proyek ini mendemonstrasikan bagaimana memanfaatkan model raksasa yang sudah dilatih (Pre-trained Model) untuk mengklasifikasikan gambar kustom menggunakan teknik *Transfer Learning* dan *Fine-Tuning* di TensorFlow/Keras.

**Fitur & Teknologi:**
- **Arsitektur ResNet50:** Menggunakan bobot *ImageNet* sebagai *feature extractor* dasar.
- **Transfer Learning (Freezing):** Membekukan seluruh layer dasar dari ResNet50 untuk mencegah *catastrophic forgetting* pada tahap awal, menghemat waktu komputasi secara signifikan.
- **Custom Classification Head:** Menambahkan `GlobalAveragePooling2D` dan `Dense(128, ReLU)` untuk memadatkan fitur, diakhiri dengan `Softmax` untuk output multi-kelas.
- **Data Pipeline dengan ImageDataGenerator:** Melakukan *loading* gambar otomatis dari direktori dengan *rescaling* (1/255) dan pembagian data otomatis (*Validation Split* 20%).
- **Advanced Fine-Tuning:** Membuka (*unfreeze*) 10 layer terakhir dari model dasar dan melatih ulang dengan *learning rate* sangat kecil (0.0001) menggunakan *optimizer Adam* untuk meningkatkan akurasi spesifik pada *dataset* target tanpa merusak pemahaman dasar model.

---

> Dibuat sebagai bagian dari pembelajaran dan dokumentasi perjalanan menjadi AI Engineer.
