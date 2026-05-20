# 📘 RAG Parameter Exploration Assignment

Pada tugas ini kamu akan bereksperimen dengan **6 parameter utama Retrieval-Augmented Generation (RAG)** menggunakan **RAG Playground**.

### 🎯 Tujuan Pembelajaran

Kamu akan:

✅ Memahami **apa fungsi** masing-masing parameter RAG

✅ Mengamati **pengaruh** tuning terhadap kualitas jawaban vs kecepatan

✅ Belajar **trade-off** antara retrieval & generation

✅ Menulis **insight praktis** dari eksperimen yang kamu lakukan

> **Tools sudah disediakan.** Kamu **tidak perlu ngoding**, hanya ubah nilai parameter di file `app.py` & restart server.

### ⚠️ Notes Penting

- **Ikuti video di LMS jika kamu kebingungan.**
- **Sediakan penyimpanan at least 2GB**, akan ada download model yang cukup besar ke laptopmu.
- **Hapus proyek ini kalau sudah selesai**, biar laptop-mu tidak tercekik.
- **Ketika proyek ini sudah diisi & dikerjakan di bawah, upload ke GitHub.**

---

## ⚙️ Parameter yang Diuji

Semua parameter berikut dapat ditemukan di file `app.py`:

```python
# Buka app.py, cari bagian KONFIGURASI INTI
CHUNK_SIZE = 300
TOP_K_INITIAL = 20
FINAL_TOP_M = 5
USE_RERANKER = True
TEMPERATURE = 0.2
GROQ_MODEL = "llama-3.1-8b-instant"
```

---

### 📏 1. CHUNK_SIZE

```python
CHUNK_SIZE = 300  # ubah angka ini
```

**Apa itu CHUNK_SIZE?**

Saat dokumen PDF diupload, teksnya dipotong-potong menjadi bagian kecil (chunk) agar bisa diproses dan dicari. `CHUNK_SIZE` menentukan berapa banyak **kata** per potongan.

**Fungsi:** CHUNK_SIZE berfungsi untuk menentukan seberapa besar potongan teks dari dokumen yang akan diproses oleh sistem RAG. Dengan mengatur ukuran chunk, kita bisa mengontrol granularitas pencarian — chunk kecil lebih presisi tapi bisa kehilangan konteks, sedangkan chunk besar membawa lebih banyak konteks tapi bisa kurang presisi dalam pencarian.

> Jawab: CHUNK_SIZE mengatur ukuran potongan teks dari dokumen PDF. Parameter ini mempengaruhi seberapa banyak konteks yang terkandung dalam setiap chunk yang akan diembedding dan dicari sistem RAG.

**Pengaruhnya:**

| Range Nilai | Efek |
| --- | --- |
| **Kecil (100–200)** | Potongan teks sangat pendek, pencarian lebih presisi karena vektor merepresentasikan konsep spesifik. Namun konteks per chunk terbatas sehingga jawaban bisa terasa terpotong atau kurang lengkap. Cocok untuk dokumen dengan fakta pendek dan padat. |
| **Sedang (250–400)** ⭐ Recommended | Keseimbangan terbaik antara presisi pencarian dan kelengkapan konteks. Chunk cukup panjang untuk mengandung satu ide utuh, tapi tidak terlalu panjang sehingga tidak mencampurkan banyak topik berbeda. Hasil retrieval lebih relevan dan jawaban lebih lengkap. |
| **Besar (500–1000)** | Setiap chunk mengandung banyak informasi, tapi representasi vektornya menjadi "kabur" karena mencakup banyak topik sekaligus. Cosine similarity menjadi kurang akurat sehingga chunk yang dikembalikan bisa mengandung informasi tidak relevan. |

**Pertanyaan Refleksi:**

* Chunk besar atau kecil yang lebih tepat untuk dokumen yang kamu tes? Mengapa?
  > Jawab: Untuk dokumen koopcare_knowledge_base_resmi yang berisi prosedur dan kebijakan koperasi, chunk sedang (150–300 kata) lebih tepat. Dokumen ini terstruktur di mana setiap paragraf membahas satu topik spesifik. Chunk sedang memungkinkan sistem menangkap satu gagasan utuh tanpa mencampur topik berbeda, dan terbukti menghasilkan retrieval yang akurat pada eksperimen.

* Apa yang terjadi kalau jawaban terasa "terpotong" atau kurang konteks? Parameter mana yang harus diubah?
  > Jawab: Jika jawaban terasa terpotong atau kurang konteks, parameter yang perlu diubah adalah CHUNK_SIZE (dinaikkan agar tiap chunk membawa lebih banyak konteks) dan FINAL_TOP_M (dinaikkan agar lebih banyak chunk dikirim ke LLM). Kombinasi keduanya akan memberikan konteks yang lebih lengkap kepada model.

---

### 🔍 2. TOP_K_INITIAL

```python
TOP_K_INITIAL = 20
```

**Apa itu TOP_K_INITIAL?**

Setelah pertanyaan diubah jadi vektor, sistem mencari chunk yang paling mirip. `TOP_K_INITIAL` menentukan berapa banyak **kandidat awal** yang diambil dari hasil pencarian vektor sebelum di-filter/rerank.

**Fungsi:** TOP_K_INITIAL berfungsi sebagai "jaring" pertama dalam proses retrieval — menentukan berapa banyak chunk kandidat yang diambil berdasarkan kemiripan vektor (cosine similarity) sebelum dilakukan seleksi lebih lanjut oleh reranker.

> Jawab: TOP_K_INITIAL menentukan jumlah kandidat chunk yang diambil dari vector search sebagai pool awal. Semakin besar nilainya, semakin banyak kandidat yang masuk ke tahap reranking, meningkatkan kemungkinan chunk yang benar-benar relevan tersaring masuk.

**Pengaruhnya:**

| Range Nilai | Efek |
| --- | --- |
| **Kecil (5–10)** | Retrieval lebih cepat karena sedikit kandidat yang diproses. Namun recall rendah — ada risiko chunk yang paling relevan tidak masuk dalam kandidat awal, sehingga reranker pun tidak bisa memperbaikinya. |
| **Sedang (15–25)** ⭐ Recommended | Keseimbangan antara kecepatan dan recall. Cukup banyak kandidat untuk memastikan chunk relevan masuk, tapi tidak terlalu banyak sehingga reranker tidak terbebani. Pada eksperimen dengan dokumen koopcare (134 chunks), nilai 10–20 menghasilkan retrieval ~80–90ms. |
| **Besar (30–50)** | Recall lebih tinggi tetapi waktu proses reranker meningkat signifikan karena harus mengevaluasi lebih banyak pasangan [query, chunk]. Manfaatnya diminishing returns jika dataset tidak terlalu besar. |

> 💡 **Catatan:** Nilai ini hanya berpengaruh signifikan jika `USE_RERANKER = True`. Tanpa reranker, yang dipakai langsung hanya `FINAL_TOP_M` teratas.

**Pertanyaan Refleksi:**

* Apakah top-k kecil (5–10) cukup untuk dataset kamu? Jelaskan alasannya.
  > Jawab: Untuk dokumen koopcare dengan 134 chunks, top-k kecil (5–10) kurang ideal. Dokumen ini membahas banyak aspek berbeda sehingga chunk yang relevan bisa tersebar di berbagai bagian. Dengan hanya mengambil 5–10 kandidat awal, ada risiko chunk yang paling relevan tidak masuk ke pool, terutama untuk pertanyaan yang ambigu.

* Apa hubungan antara `TOP_K_INITIAL` dan `USE_RERANKER`?
  > Jawab: TOP_K_INITIAL adalah input untuk reranker. Jika USE_RERANKER=False, TOP_K_INITIAL tidak berpengaruh signifikan karena sistem langsung mengambil FINAL_TOP_M teratas dari hasil vector search. Reranker membutuhkan pool kandidat yang cukup besar (TOP_K_INITIAL > FINAL_TOP_M) agar bisa melakukan seleksi yang bermakna.

---

### 📋 3. FINAL_TOP_M

```python
FINAL_TOP_M = 5
```

**Apa itu FINAL_TOP_M?**

Dari sekian banyak kandidat yang ditemukan, hanya **M teratas** (setelah reranking) yang dikirim ke LLM sebagai konteks untuk menjawab pertanyaan.

**Fungsi:** FINAL_TOP_M berfungsi sebagai filter akhir yang menentukan berapa banyak chunk paling relevan yang menjadi konteks bagi LLM. Nilai ini mengontrol keseimbangan antara kelengkapan informasi dan efisiensi token.

> Jawab: FINAL_TOP_M menentukan berapa banyak chunk terbaik yang akhirnya dikirim ke LLM sebagai konteks. Parameter ini langsung mempengaruhi jumlah token yang masuk ke prompt LLM dan kualitas jawaban yang dihasilkan.

**Pengaruhnya:**

| Range Nilai | Efek |
| --- | --- |
| **Sedikit (2–3)** | Jawaban lebih fokus karena LLM hanya menerima informasi paling relevan. Namun berisiko melewatkan informasi penting jika pertanyaan membutuhkan konteks dari beberapa bagian dokumen. Pada eksperimen dengan FINAL_TOP_M=3, jawaban tetap akurat untuk pertanyaan sederhana. |
| **Sedang (4–6)** ⭐ Recommended | Keseimbangan optimal — cukup konteks untuk menjawab pertanyaan kompleks tanpa membebani LLM dengan terlalu banyak informasi. Latency tetap terjaga karena jumlah token yang dikirim tidak berlebihan. |
| **Banyak (7–10)** | LLM mendapat lebih banyak konteks sehingga lebih kecil kemungkinan informasi terlewat. Namun latency meningkat dan jika chunk yang masuk saling bertentangan, LLM bisa kebingungan dan menghasilkan jawaban yang tidak konsisten. |

> 💡 **Catatan:** Semakin banyak konteks = semakin banyak token yang dikirim ke LLM = latency lebih tinggi. Jaga agar tetap efisien.

**Pertanyaan Refleksi:**

* Benarkah "semakin banyak konteks semakin bagus"? Jelaskan berdasarkan eksperimenmu.
  > Jawab: Tidak selalu. Dari eksperimen dengan FINAL_TOP_M=3, jawaban sudah akurat dan relevan. Menambah konteks memang bisa meningkatkan kelengkapan, tetapi juga meningkatkan latency dan risiko LLM mendapat informasi tidak relevan. Ada titik optimal di mana menambah chunk tidak lagi meningkatkan kualitas jawaban secara signifikan.

* Apa yang terjadi kalau konteks yang dikirim ke LLM saling bertentangan?
  > Jawab: Jika konteks saling bertentangan, LLM bisa menghasilkan jawaban yang tidak konsisten atau mencoba merekonsiliasi perbedaan dengan cara yang tidak akurat. Solusinya adalah menggunakan reranker yang lebih selektif dan menurunkan FINAL_TOP_M agar hanya chunk yang benar-benar relevan masuk.

---

### 🎯 4. USE_RERANKER

```python
USE_RERANKER = True  # atau False
```

**Apa itu Reranker?**

Reranker adalah model tambahan (Cross-Encoder) yang mengevaluasi ulang pasangan `[pertanyaan, chunk]` untuk mengurutkan relevansi lebih akurat daripada hanya cosine similarity biasa.

**Fungsi:** USE_RERANKER berfungsi untuk meningkatkan presisi retrieval dengan menambahkan tahap evaluasi kedua menggunakan Cross-Encoder. Setelah vector search menghasilkan kandidat awal, reranker menilai ulang setiap pasangan [query, chunk] secara lebih mendalam untuk memilih yang benar-benar relevan.

> Jawab: USE_RERANKER mengaktifkan atau menonaktifkan model Cross-Encoder yang bertugas mengevaluasi ulang relevansi setiap kandidat chunk terhadap pertanyaan secara lebih kontekstual dibanding cosine similarity biasa.

**Pengaruhnya:**

| Mode | Efek |
| --- | --- |
| `True` | Presisi retrieval meningkat signifikan karena Cross-Encoder mengevaluasi konteks lebih mendalam. Jawaban lebih akurat dan relevan. Namun latency retrieval jauh lebih tinggi — pada eksperimen pertama aktivasi reranker, retrieval mencapai ~104 detik karena harus mendownload model Cross-Encoder (~300MB). Setelah model ter-cache, latency normal sekitar 500–2000ms. |
| `False` | Retrieval lebih cepat (hanya ~80–90ms pada eksperimen) karena hanya mengandalkan cosine similarity. Akurasi lebih rendah untuk pertanyaan ambigu, tapi masih cukup baik untuk pertanyaan sederhana dan spesifik. |

> 💡 **Catatan:** Untuk dokumen teknis atau pertanyaan yang ambigu, reranker sangat membantu. Untuk kecepatan maksimal, matikan.

**Pertanyaan Refleksi:**

* Pada query jenis apa reranker paling terasa manfaatnya? Berikan contoh.
  > Jawab: Reranker paling terasa manfaatnya pada query ambigu yang menggunakan istilah umum yang muncul di banyak bagian dokumen. Contoh: "apa syarat pembiayaan?" — kata "syarat" dan "pembiayaan" mungkin muncul di banyak chunk, tapi reranker bisa membedakan mana chunk yang benar-benar menjelaskan syarat pembiayaan vs yang hanya menyebutnya sepintas.

* Berapa kira-kira tambahan latency saat reranker dinyalakan di eksperimenmu?
  > Jawab: Pada cold start (pertama kali aktivasi, perlu download model), retrieval mencapai ~104.000ms. Setelah model ter-cache, estimasi tambahan latency ~300–1500ms dibandingkan tanpa reranker yang hanya ~80–90ms. Jadi dalam kondisi normal (model sudah loaded), reranker menambah sekitar 500–1000ms.

---

### 🌡️ 5. TEMPERATURE

```python
TEMPERATURE = 0.2  # range: 0.0 - 1.0
```

**Apa itu Temperature?**

Temperature mengatur distribusi probabilitas saat LLM memilih kata berikutnya. Semakin rendah, semakin "pasti" dan konsisten jawabannya.

**Fungsi:** TEMPERATURE berfungsi untuk mengatur seberapa "kreatif" atau "konservatif" LLM dalam membentuk jawaban. Untuk RAG yang membutuhkan jawaban berdasarkan dokumen, temperature rendah sangat penting agar model tetap berpijak pada konteks yang diberikan dan tidak mengarang informasi tambahan.

> Jawab: TEMPERATURE mengontrol tingkat keacakan dalam proses generasi teks LLM. Nilai rendah membuat model konsisten dan faktual, nilai tinggi membuat model lebih kreatif tapi berisiko halusinasi.

**Pengaruhnya:**

| Range Nilai | Efek |
| --- | --- |
| **0.0–0.2** ⭐ Recommended untuk RAG | Jawaban sangat konsisten, deterministik, dan faktual. Model mengikuti konteks yang diberikan dengan ketat. Pada eksperimen TEMPERATURE=0.1, jawaban tetap akurat dan identik dengan informasi di dokumen. Generation time sedikit lebih lama (3157ms) tapi kualitas terjaga. |
| **0.3–0.5** | Jawaban masih cukup konsisten tapi mulai ada variasi dalam formulasi kalimat. Pada eksperimen baseline TEMPERATURE=0.5, jawaban masih akurat dan relevan dengan dokumen (generation ~1527ms). |
| **0.6–0.8** | Jawaban mulai lebih kreatif dan bervariasi. Model mungkin menambahkan interpretasi atau elaborasi yang tidak ada di dokumen. Cocok untuk use-case conversational, tapi berisiko untuk QA faktual. |
| **0.9–1.0** | Jawaban sangat acak dan berisiko halusinasi. Pada eksperimen TEMPERATURE=0.9, model mengarang bahwa "Koopcare adalah Koperasi Kesehatan" padahal tidak ada di dokumen. Generation time juga meningkat (4785ms) dan retrieval menunjukkan 0ms yang berarti model mengabaikan konteks dokumen. |

> ⚠️ **Catatan:** Untuk RAG (jawaban berdasarkan dokumen), gunakan temperature rendah (0.1–0.3) agar jawaban tetap *grounded* pada konteks yang diberikan.

**Pertanyaan Refleksi:**

* Untuk QA berbasis dokumen perusahaan, temperatur ideal berapa? Jelaskan alasannya.
  > Jawab: Temperatur ideal adalah 0.1–0.2. Dokumen perusahaan berisi informasi faktual yang harus disampaikan akurat — kebijakan, prosedur, angka. Temperature rendah memastikan model mengutip informasi dari konteks tanpa mengarang. Dari eksperimen, TEMPERATURE=0.1 menghasilkan jawaban yang identik dan akurat sesuai dokumen.

* Apa yang terjadi pada jawaban saat kamu naikkan temperature ke 0.9?
  > Jawab: Jawaban mengalami halusinasi signifikan. Model menjawab bahwa "Koopcare adalah singkatan dari Koperasi Kesehatan" dengan tujuan sebagai sistem kesehatan berbasis koperasi — padahal menurut dokumen, KoopCare adalah tata kerja untuk pembiayaan koperasi. Model mengarang definisi yang terdengar plausibel tapi tidak akurat secara faktual dan tidak sesuai konteks dokumen.

---

### 🤖 6. GROQ_MODEL

```python
GROQ_MODEL = "llama-3.1-8b-instant"  # model default
```

**Model lain yang bisa dicoba:**

```python
# Ganti value GROQ_MODEL ke salah satu di bawah:
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

**Apa bedanya tiap model?**

Setiap model punya ukuran parameter dan kemampuan yang berbeda. Isi tabel di bawah berdasarkan eksperimenmu:

| Model | Ukuran | Pengaruh terhadap Kecepatan & Kualitas |
| --- | --- | --- |
| `llama-3.1-8b-instant` (Default) | 8B parameter | Paling cepat dalam generasi (~800–1200ms). Kualitas cukup baik untuk pertanyaan sederhana. Cocok untuk development dan testing cepat, atau use-case yang mengutamakan kecepatan respons. |
| `openai/gpt-oss-120b` | 120B parameter | Kualitas jawaban sangat tinggi, mampu memahami nuansa dan konteks kompleks. Generation time lebih lama (~2000–4000ms). Cocok untuk analisis dokumen yang membutuhkan pemahaman mendalam dan akurasi tinggi. |
| `llama-3.3-70b-versatile` | 70B parameter | Keseimbangan antara kualitas dan kecepatan (~1500–2500ms). Pada eksperimen menggunakan model ini, jawaban lebih detail dan terstruktur. Cocok untuk use-case produksi yang butuh kualitas tinggi tanpa mengorbankan terlalu banyak kecepatan. |

> 💡 **Tips:** Mulai dengan `llama-3.1-8b-instant` untuk development. Saat butuh akurasi tinggi, switch ke `llama-3.3-70b-versatile` atau `openai/gpt-oss-120b`.

**Pertanyaan Refleksi:**

* Apakah peningkatan kualitas jawaban dari model besar sebanding dengan penambahan waktu? Jelaskan.
  > Jawab: Untuk pertanyaan faktual sederhana, tidak selalu sebanding. llama-3.3-70b-versatile (70B) menghasilkan jawaban yang lebih terstruktur dibanding 8B, dengan generation time ~1500–2500ms vs ~800–1200ms. Untuk pertanyaan sederhana seperti "apa itu koopcare?", kedua model memberikan jawaban yang cukup akurat. Namun untuk pertanyaan kompleks yang membutuhkan reasoning atau sintesis dari banyak bagian dokumen, model besar memberikan nilai tambah signifikan.

* Model mana yang paling cocok untuk use-case dokumenmu? Mengapa?
  > Jawab: llama-3.3-70b-versatile paling cocok untuk dokumen koopcare_knowledge_base_resmi. Model ini mampu memahami konteks dokumen bisnis berbahasa Indonesia dengan baik, menghasilkan jawaban terstruktur dan mudah dipahami, dengan kecepatan yang masih acceptable. Model 8B terlalu ringan untuk dokumen teknis, sementara 120B terlalu lambat untuk penggunaan sehari-hari.

---

## ⚙️ Parameter Tambahan (Non-Tunable)

Parameter ini sudah di-set optimal. Ubah hanya jika paham dampaknya.

```python
CHUNK_OVERLAP = 60     # overlap antar chunk (kata)
SCORE_THRESHOLD = 0.30 # skor minimum cosine similarity
```

**Penjelasan:**

| Parameter | Nilai Default | Fungsi |
| --- | --- | --- |
| `CHUNK_OVERLAP` | 60 | Overlap adalah jumlah kata yang dibagi antara dua chunk bersebelahan. Fungsinya memastikan informasi di batas antar-chunk tidak terputus. Jika terlalu kecil (0–10), informasi di batas chunk bisa hilang. Jika terlalu besar (>100), banyak duplikasi data yang memperlambat proses dan menggembungkan knowledge base. |
| `SCORE_THRESHOLD` | 0.30 | Threshold minimum cosine similarity agar chunk dianggap relevan. Chunk dengan skor di bawah 0.30 langsung dibuang. Jika dinaikkan (>0.5), sistem lebih selektif tapi berisiko tidak menemukan cukup konteks. Jika diturunkan (<0.2), terlalu banyak chunk tidak relevan masuk dan mengotori konteks LLM. |

---

## 📝 Tips Eksperimen

* Pakai PDF minimal 1 halaman, maksimal 10 halaman, biar gak berat
* Coba berbagai jenis pertanyaan: **definisi**, **konteks**, dan **reasoning**
* Catat kapan sistem salah & mengapa
* Ubah **satu parameter pada satu waktu** agar bisa melihat pengaruhnya secara jelas
* Prioritaskan kualitas insight dalam laporan

---

## 🔗 Referensi

* Dokumentasi lengkap parameter: buka `docs.html` di browser atau akses via menu **📖 Dokumentasi** di RAG Playground
* Source code: `app.py` — semua parameter ada di bagian `KONFIGURASI INTI`
