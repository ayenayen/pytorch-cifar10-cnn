# CIFAR-10 Image Classification with PyTorch

Proyek ini adalah implementasi dasar *Convolutional Neural Network* (CNN) menggunakan PyTorch untuk mengklasifikasikan gambar dari dataset CIFAR-10. Proyek ini dibangun sebagai bagian dari portofolio *Deep Learning* dasar.

## Deskripsi
Dataset CIFAR-10 terdiri dari 60.000 gambar berwarna dengan ukuran 32x32 piksel, yang terbagi dalam 10 kelas (pesawat, mobil, burung, kucing, rusa, anjing, katak, kuda, kapal, truk). 
Model CNN yang dibuat memiliki arsitektur sebagai berikut:
- 2 Layer Konvolusi (`Conv2d`)
- Layer MaxPooling (`MaxPool2d`)
- Fungsi Aktivasi ReLU
- 2 Layer Fully Connected (`Linear`)

## Hasil Eksperimen
Model dilatih selama 5 iterasi (Epoch) menggunakan *Optimizer Adam* dan *CrossEntropyLoss*.
- **Akurasi Testing:** ~71% pada 10.000 gambar yang belum pernah dilihat oleh model (Test set).
- Hardware yang digunakan: GPU (CUDA).

## Struktur Kode
- `pytorch.ipynb`: Berisi seluruh langkah eksperimen mulai dari *preprocessing* (transform), definisi model CNN, *training loop*, evaluasi, hingga visualisasi perbandingan prediksi vs nilai asli.
