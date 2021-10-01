# Book Recommendation System

## Domain Proyek
Data dari [Picodi.com](https://www.picodi.com/id/mencari-penawaran/pembelian-buku-di-indonesia-dan-di-seluruh-dunia) mengatakan, jumlah permintaan buku di pasar Indonesia masih cukup tinggi. Di bulan Desember 2018 jumlahnya bisa mencapai 12 persen dari transaksi tahunan, atau meningkat hampir dua kali lipat dari penjualan di awal tahun.

Sebanyak 73% responden mendapatkan buku dari toko buku dan 55% dari toko online. Banyak toko buku online maupun offline yang saling bersaing untuk memberikan kepuasan lebih kepada pelanggannya dengan mempertimbangkan berbagai atribut. Sistem rekomendasi adalah teknologi powerful yang dapat membantu pelanggan menemukan barang yang ingin mereka beli dengan lebih mudah dan cepat. Sistem rekomendasi juga menjadi salah satu alat terkuat untuk meningkatkan keuntungan dan mempertahankan pembeli [(P. Mathew, 2016)](https://ieeexplore.ieee.org/abstract/document/7684166).

## Business Understanding
### Problem Statement
- Bagaimana cara mengolah data buku, pengguna, dan rating untuk digunakan sebagai informasi sistem rekomendasi?
- Bagaimana cara merekomendasikan buku dengan teknik machine learning?

### Goals
Tujuan dari proyek ini adalah membuat pendekatan sistem rekomendasi dengan machine learning untuk memberikan rekomendasi buku kepada pelanggan berdasarkan data buku, pengguna, dan rating.

### Solution Statements
Ada 3 solusi yang diberikan bergantung dengan hasil rekomendasi yang ingin dicapai dan data yang dimiliki, yaitu :
- **Popular recommendation**. Merekomendasikan buku berdasarkan rata-rata rating dan jumlah rating yang diterima per buku dengan membuat weighted rating. Teknik ini dapat berguna untuk merekomendasikan buku kepada seluruh pengguna baik yang belum memiliki riwayat transaksi maupun yang sudah.
- **Content-Based Recommendation**. Merekomendasikan buku berdasarkan kemiripan Author buku kepada pengguna yang melakukan transaksi pada buku tersebut. Kelebihan teknik ini tidak membutuhkan data transaksi dari user lain karena rekomendasi spesifik untuk user tersebut. Sehingga mempermudah merekomendasikan buku pada jumlah pengguna yang sangat besar. Sedangkan kekurangannya, rekomendasi hanya terbatas pada interest dari pengguna dan tidak bisa memperluas interest tersebut. Selain itu, diperlukan fitur buku yang lebih bagus selain Author agar hasil rekomendasi semakin baik.
- **Model-Based Collaborative filtering Recommendation**. Merekomendasikan buku berdasarkan riwayat transaksi (rating) pengguna untuk memprediksi dan menghitung rating yang akan diberikan pengguna pada buku lain menggunakan model machine learning SVD. Algoritma berbasis matrix factorization ini dipopulerkan oleh Simon Funk pada Netflix Prize. Kekurangan dari collaborative filtering tidak bisa merekomendasikan item yang tidak memiliki riwayat transaksi.

## Data Understanding
Dataset diambil dari [Kaggle Book Recommendation Dataset](https://www.kaggle.com/arashnic/book-recommendation-dataset). Dataset berisi 3 file csv, yaitu:
- **Users**. Berisi informasi user
  - UserID: identitas unik user berupa integer agar user anonymous
  - Location: lokasi tempat tinggal user
  - Age: umur user
- **Books**. Berisi informasi buku
  - Book-Title: judul
  - Book-Author: penulis
  - Year-Of-Publication: Tahun terbit
  - Publisher: Penerbit
  - Image-URL-S, Image-URL-M, Image-URL-L: link sampul buku
- **Ratings**. Berisi informasi rating buku dari user
  - Book-Rating: rating buku

## Data Preparation
Data preparation adalah tahap di mana kita melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. Ada beberapa tahapan yang dilakukan pada data preparation, antara lain :
- **Handling Missing Value**. Proses mengolah missing value (ex: data umur yang null, data rating 0) dengan menghapus atau mengganti data tersebut dengan value lain.
- **Encoding**. Melakukan encoding data UserID dan Book Title agar dapat dibaca model dengan baik.
- **Transformasi Data**. Mentransformasikan book author menjadi matrix dengan TF-IDF Vectorizer untuk mengidentifikasi korelasi antara buku dan authornya.
- **Merge Data**. Menggabungkan data rating dan buku untuk dijadikan dataset.

## Modelling
- **Popular Recommendation**.
Mencari informasi rata-rata rating dan jumlah rating yang diterima per buku. Untuk menggabungkan kedua informasi tersebut, dibuatlah weighted rating, kemudian memilih 20 produk terbaik berdasarkan weighted rating. Weighted Rating = (Rv + Cm) / (v + m), dimana v adalah jumlah rating diterima per buku, R adalah rata-rata rating per buku, C adalah rata-rata rating seluruh buku, dan m adalah minimal jumlah rating yang diterima.
  ![Weighted Rating](https://miro.medium.com/max/368/1*fGziZl2Do-VyQXSCPq_Y2Q.png)

- **Content-Based Recommendation**
Menghitung derajat kesamaan (similariy degree) antar buku berdasarkan author dengan teknik cosine similarity dari library scikit learn. Cosine similarity menghitung kesamaan sebagai dot product yang dinormalisasi dari X dan Y: 
  ![Cosine Similarity](https://wikimedia.org/api/rest_v1/media/math/render/svg/1d94e5903f7936d3c131e040ef2c51b473dd071d)

- **Model-Based Collaborative Filtering Recomendation**
Mentraining data user buku dengan model SVD dari library Surprise kemudian mengurutkan 10 buku dengan prediksi rating tertinggi. 

## Evaluation
Hasil evaluasi dari model SVD menggunakan metode k-fold cross validation. Metode ini adalah salah satu dari jenis pengujian cross validation yang berfungsi untuk menilai kinerja proses sebuah metode algoritme dengan membagi sampel data secara acak dan mengelompokkan data tersebut sebanyak nilai K fold. Dimana data training adalah K-1 fold dan sisanya digunakan sebagai data testing. Kemudian hasil testing dihitung dengan matriks:
- **Mean Absolute Error (MAE)**, merepresentasikan rata-rata perbedaan mutlak antara nilai aktual dan prediksi pada dataset. MAE mengukur rata-rata residu dalam dataset. MAE lebih intuitif dalam memberikan rata-rata error dari keseluruhan data.
  ![MAE](https://1.bp.blogspot.com/-OY4iwFkwEdQ/X8J8nmJFPFI/AAAAAAAACYo/hFjo4vbDdWguXH5XKhHEXWihbKKIkZA_wCLcBGAsYHQ/s241/Rumus%2BMAE.jpg)
- **Root Mean Squared Error (RMSE)**, dihitung dengan mengkuadratkan error (prediksi – observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan.
  ![RMSE](https://1.bp.blogspot.com/-1Dkr12kqVyc/Xa60hhWWcTI/AAAAAAAABIE/-xUrC9kWrWM-VMUq2PIWQ5_v51xNWBnWwCLcBGAsYHQ/s1600/Picture4.jpg)













 
