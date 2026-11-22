# Lari Balok CV

Prototipe **wasit bantu berbasis computer vision** untuk olahraga tradisional lari balok
(Tugas Besar FK2052 Olahraga Tradisional, Kelompok 19).

Satu kamera dipasang di samping lintasan. Gambarnya diproses oleh YOLO11-pose + ByteTrack, lalu
aturan permainan menentukan pelanggaran, urutan juara, dan klip bukti. Semua hasil tampil di
dashboard browser untuk wasit. Sistem ini hanya alat bantu: wasit tetap berwenang memutuskan.

## Fitur

- **Pelanggaran**: kaki, lutut, tangan, atau pinggul yang menyentuh tanah selama ≥ 100 ms membuat
  pemain berstatus GUGUR, disertai bunyi peringatan dan klip bukti.
- **Finis**: dihitung saat pergelangan kaki pertama melewati garis FINIS. Waktunya diinterpolasi
  antar-frame. Dua pemain yang selisihnya kurang dari 1 frame dinyatakan SERI.
- **Wewenang wasit**: klip bisa ditinjau per frame, lalu disahkan atau dibatalkan. Setelah itu
  peringkat langsung dihitung ulang.
- **Aba-aba start**: hitung mundur 3-2-1 dengan bunyi.
- **Rekaman**: setiap lomba menyimpan video mentah dan `result.json` sebagai bahan evaluasi.
- **Sumber video**: kamera USB secara live, atau file video.

## Instalasi (Windows, Python 3.12)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # GPU NVIDIA
pip install -r requirements.txt
python tests/test_judge.py                                                         # uji aturan
```

Laptop tanpa GPU NVIDIA: pasang `pip install torch torchvision` (versi CPU), lalu jalankan dengan
`--model yolo11n-pose.pt --imgsz 640`. Bobot YOLO terunduh otomatis saat aplikasi pertama kali dijalankan.

## Menjalankan

```powershell
python app.py --source 0                # kamera indeks 0 (kamera USB eksternal biasanya 1)
python app.py --source samples\uji.mp4  # file video, diputar dengan kecepatan asli
```

Setelah itu buka **http://localhost:8000**. Untuk proyektor, tekan F11 agar layar penuh.

## Prosedur di hari lomba

1. **Pasang kamera.** Letakkan kamera di tripod sesuai bagian [penempatan kamera](#penempatan-kamera-dan-arena),
   sambungkan ke laptop, lalu jalankan aplikasi.
2. **Kalibrasi**, cukup sekali untuk setiap posisi kamera. Tekan **Kalibrasi**, lalu:
   - Klik 4 ujung lakban di lantai dengan urutan START dekat, START jauh, FINIS dekat, FINIS jauh.
   - Isi jumlah lintasan, panjang lintasan, dan tinggi balok.
   - Tekan **Ambil warna balok**, lalu klik sisi balok.
   - Periksa pratinjau, lalu tekan **Simpan**.

   Di pratinjau, pita berwarna menandai lintasan dan magenta menandai piksel yang dikenali sebagai
   balok. Kalau ada benda lain yang ikut magenta, ganti warna balok atau singkirkan benda itu.
3. **Siapkan pemain.** Pemain berdiri di atas balok di belakang garis START. Kartu lintasan
   menunjukkan SIAP saat pemain sudah terdeteksi.
4. **Mulai.** Tekan **Mulai**; hitung mundur 3-2-1 berjalan dan lomba dimulai saat bunyi MULAI.
5. **Pelanggaran** ditandai otomatis dengan status GUGUR dan bunyi. Wasit yang memutuskan apakah
   pemain dihentikan. Selama itu sistem tetap mencatat waktu finisnya.
6. **Hasil.** Peringkat muncul begitu pemain finis, dan podium tampil saat semua pemain sah sudah
   finis. Jika ada keberatan, tekan **Tinjau** lalu **Sahkan** atau **Batalkan**.
7. **Lomba berikutnya.** Tekan **Reset**. Untuk sumber berupa file video, video diputar ulang dari awal.

## Aturan yang diterapkan

Semua parameter ada di bagian atas `judge.py`.

| Keputusan | Aturan di sistem |
|---|---|
| Pelanggaran | Titik pergelangan kaki, lutut, pergelangan tangan, atau pinggul dianggap menyentuh tanah bila memenuhi empat syarat. **(a)** Titik terlihat (confidence ≥ `KP_CONF`); titik yang tertutup tidak dinilai. **(b)** Tingginya dari garis tanah lintasan kurang dari ½ tinggi balok ditambah tinggi sendi itu dari pijakannya. **(c)** Titik tidak berada di atas balok milik pemain itu (tangan boleh memegang sisi balok). **(d)** Titik diam, yaitu berpindah kurang dari `STILL_CM` = 6 cm. Kondisi ini harus bertahan ≥ `MIN_CONTACT_S` = 100 ms (3 frame pada 30 fps). |
| Finis | Pergelangan kaki pertama melewati garis FINIS. Lintasan garis dikonfirmasi pada 2 frame berturut-turut (`FINISH_FRAMES`), dan waktunya diinterpolasi antar-frame. |
| Setelah finis | Sentuhan yang mulai sesudah kaki melewati garis diabaikan. Sentuhan yang mulai sebelumnya tetap dihitung. |
| Juara | Pemain tanpa pelanggaran aktif, diurutkan menurut waktu finis. Selisih kurang dari 1 frame dinyatakan seri dan perlu ditinjau. |
| Identitas pemain | ID ByteTrack dipetakan ke lintasan berdasarkan posisi kaki. Pemain tidak berpindah lintasan, jadi pertukaran ID tidak mengubah hasil. |
| Wewenang wasit | Sistem tidak menghentikan pemain. Waktu finis pemain yang GUGUR tetap dicatat, sehingga pembatalan pelanggaran bisa memulihkan peringkatnya. |

**Perbedaan dari slide:**
- Finis memakai pergelangan kaki, bukan titik tubuh terdepan, karena tangan atau kepala yang
  menjulur bisa memicu finis lebih awal.
- Ada tambahan syarat *diam*, supaya kaki yang sedang melangkah tidak ikut ditandai.
- Pinggul ikut dinilai untuk menangkap kasus jatuh terduduk.
- Kalibrasi cukup 4 titik; garis tanah tiap lintasan dihitung lewat homografi.
- Rompi tidak dibutuhkan algoritma, tetapi tetap berguna untuk wasit.

## Penempatan kamera dan arena

- **Posisi kamera**: di samping lintasan, tegak lurus terhadapnya, di tengah panjang lintasan,
  setinggi ±2 m dan sedikit menunduk. Kedua ujung lakban START dan FINIS harus masuk frame.
- **Jarak antar-lintasan ≥ 1 m**. Jika lebih rapat, kaki di atas balok di lintasan dekat bisa
  terbaca sebagai milik lintasan jauh.
- **Panjang lintasan 5–10 m** untuk satu kamera. Lintasan yang lebih panjang membuat pemain terlalu kecil di gambar.
- **Balok**: sisinya diberi warna mencolok, misalnya lakban hijau, yang tidak dipakai di rompi,
  sepatu, atau latar. Balok yang lebih tinggi (≥ 12 cm) membuat "di atas balok" lebih mudah
  dibedakan dari "di tanah".
- **Rompi**: samakan dengan warna lintasan di layar, yaitu 1 merah, 2 biru, 3 kuning.
  Lintasan 1 adalah yang paling jauh dari kamera.
- **Video**: 1080p 30 fps, cahaya merata, dan area di belakang lintasan bebas dari penonton.

## Merekam video uji

Rekaman diperlukan untuk menyetel ambang dan menghitung metrik di slide "Data & Evaluasi".

- **Cara merekam**: gunakan penempatan kamera seperti di atas. Bisa lewat aplikasi (setiap lomba
  tersimpan di `runs/<waktu>/race.mp4`), atau dengan HP di tripod pada 1080p 30 fps tanpa
  menggerakkan kamera dan tanpa zoom.
- **Skenario yang direkam**:
  - lomba sah dengan 2 dan 3 pemain;
  - pelanggaran disengaja: kaki menapak tanah sebentar (±0,2 s) dan lama, tangan menopang di
    tanah, lutut menyentuh tanah, dan jatuh;
  - finis yang berdekatan;
  - pemain yang saling menutupi;
  - dua kondisi cahaya.
- **Memutar ulang**: rekaman bisa dijalankan lagi dengan
  `python app.py --source runs\<waktu>\race.mp4`. Kalibrasi sekali untuk video tersebut.
- **Penyimpanan**: simpan video uji di folder `samples/`. Folder ini tidak di-commit karena repo
  publik dan video memuat wajah pemain.

## Hasil per lomba

Setiap lomba menghasilkan folder `runs/<tanggal_jam>/` berisi:

- `race.mp4`: video mentah lomba;
- `result.json`: status, waktu finis, peringkat, semua kejadian, keputusan wasit, dan ID track per
  lintasan (untuk menghitung pertukaran identitas);
- `clips/<id>/`: frame beranotasi ±3 detik di sekitar setiap kejadian.

## Batasan yang diketahui

- **Satu kamera.** Kaki yang tertutup oleh pemain di lintasan dekat tidak dinilai. Kartu lintasan
  menampilkan TERTUTUP agar wasit tahu.
- **Ambang belum disetel.** Ambang tinggi, jarak, dan waktu belum disetel dengan rekaman lari
  balok sungguhan.
- **Deteksi balok berbasis warna.** Cara ini peka terhadap cahaya dan benda yang warnanya sama.
  Tahap berikutnya: melatih detektor YOLO kelas "balok" dari rekaman. Label awal bisa dibuat
  otomatis dengan deteksi warna, lalu tinggal dikoreksi.
- **Durasi rekaman.** `race.mp4` ditulis pada fps nominal. Jika laptop menjatuhkan frame, durasi
  video bisa sedikit lebih pendek dari durasi lomba. Waktu resmi tetap diambil dari timestamp frame.
- **Belum tersedia:** kamera kedua, deteksi false start, dan penilaian teknik memindah balok.
