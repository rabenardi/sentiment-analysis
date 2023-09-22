Fachry, 2023.

### Analisis Sentimen
Dalam repo ini berisi sebuah program yang menggunakan Keras berbasis Tensorflow untuk melakukan analisis sentimen.

Analisis sentimen adalah penggunaan NLP (Natural Language Processing) untuk mengambil nilai subjektif dan takaran baik-buruknya suatu kalimat. Analisis sentimen juga disebut dengan "opinion mining" atau "emotion AI".

---
### Instalasi
Dengan anggapan bahwa program `git` dan `python3` telah terpasang.

Pertama-tama, clone repo ini
```
git clone https://github.com/rabenardi/sentimental-analysis.git
```

Lalu, install paket yang dibutuhkan dengan menjalankan
```
pip3 install -r requirements.txt
```

Sebelum bisa dipakai, buat berkas enviroment dengan notepad (atau penyunting teks lainnya), lalu tambahkan nilai berikut (awas, besar-kecilnya huruf diperhatikan)
```
MAX_VOCAB=
MAX_LEN=
EPOCHS=
LAYERS_DENSE=
```
(nilai bisa diatur sesuka hati)

---
### Melatih model
Jika kalian ingin melatih modelnya dari awal, kalian bisa jalankan
```
python train.py
```
untuk melatih model dengan sampel bawaan.

Jika kalian ingin menambahkan sampel lain, pastikan (1) berupa csv, (2) kalian tau lokasi sampelnya di mana (_path_-nya) dan (3) apa kolom sentimen dan teksnya. Misal ada sampel dengan kolom
```
,full_text,date,polarity,subjectivity,analysis
0,manfaat penuh eh udah kasih admin fee yaudah fokus cimb niaga on account fasilitas on par substitusi,Thu Nov 12 17:27:57 +0000 2020,0.425,0.525,positive
```
Maka kolom sentimennya adalah `analysis` dan kolom teksnya dalah `full_text`.

Jika sudah teridentifikasi, maka mulai tambahkan sampel tersebut ke kode dengan menambahkan/meng-_append_ entri (berbentuk `dict`) berikut ke dalam variable _other_samples_ di `sources.py`.
```
other_samples = [{
    "the_path": lokasi,
    "sentiment_col": kolom_sentimennya,
    "text_col": kolom_teksnya
}, ...]
```

---
### Pengujian
Untuk mulai menguji sentimen suatu kalimat, kalian dapat menggunakan fungsi `predict_sentiment` di `model_test.py`.

Untuk versi interaktifnya, jalankan
```
python .
```

