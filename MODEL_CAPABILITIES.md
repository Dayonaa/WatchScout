# 🧠 Panduan & Katalog Kapabilitas Model Google GenAI (Gemini API)

Dokumen ini memetakan seluruh model yang tersedia pada akun/API key Anda beserta fungsi, limit token, fitur unggulan, dan rekomendasi penggunaannya.

---

## 📑 Daftar Isi
1. [Model Flagship & General Reasoning (Gemini Flash & Pro)](#1-model-flagship--general-reasoning)
2. [Pembuatan Video AI (Google Veo)](#2-pembuatan-video-ai-google-veo)
3. [Pembuatan & Editing Gambar (Imagen / Nano Banana)](#3-pembuatan--editing-gambar-nano-banana)
4. [Audio, Text-to-Speech (TTS) & Transkripsi](#4-audio-text-to-speech-tts--transkripsi)
5. [Realtime Voice & Multimodal Streaming (Live API)](#5-realtime-voice--multimodal-streaming-live-api)
6. [Generasi Musik & Audio Efek (Lyria)](#6-generasi-musik--audio-efek-lyria)
7. [AI Agent, Coding & Computer Use (Antigravity & Tool Use)](#7-ai-agent-coding--computer-use)
8. [Deep Research & Fakta Teratribusi (Deep Research & AQA)](#8-deep-research--fakta-teratribusi)
9. [Embeddings & Vector Search (RAG)](#9-embeddings--vector-search-rag)
10. [Open Weights LLM (Gemma)](#10-open-weights-llm-gemma)
11. [Rangkuman Rekomendasi Pemilihan Model](#11-rangkuman-rekomendasi-pemilihan-model)

---

## 1. Model Flagship & General Reasoning
Model serbaguna untuk teks, penalaran logika tingkat tinggi, pemahaman dokumen panjang, coding, dan analisis multimodal (gambar/video/audio).

| Model ID | Display Name | Input Limit | Output Limit | Kegunaan Utama |
| :--- | :--- | :--- | :--- | :--- |
| `gemini-3.8-flash` | **Gemini 3.8 Flash** | 1,048,576 | 65,536 | **Paling direkomendasikan** untuk sebagian besar tugas. Sangat cepat, hemat biaya, dan mendukung context window 1 juta token. |
| `gemini-3.7-flash` | Gemini 3.7 Flash | 1,048,576 | 65,536 | Model generasi sebelumnya dengan stabilitas tinggi dan penalaran fleksibel. |
| `gemini-3.6-flash` | Gemini 3.6 Flash | 1,048,576 | 65,536 | Varian Flash cepat untuk throughput tinggi. |
| `gemini-3.5-flash` | Gemini 3.5 Flash | 1,048,576 | 65,536 | Model efisien untuk pipeline pemrosesan teks berkecepatan tinggi. |
| `gemini-3.5-flash-lite` | Gemini 3.5 Flash Lite | 1,048,576 | 65,536 | Varian super ringan untuk tugas klasifikasi teks dan ekstraksi data berbiaya paling minimal. |
| `gemini-3.1-pro-preview` | Gemini 3.1 Pro Preview | 1,048,576 | 65,536 | Penalaran analitis kompleks, pemecahan masalah matematika/sains, dan arsitektur kode berskala besar. |
| `gemini-3.1-pro-preview-customtools` | Gemini 3.1 Pro Custom Tools | 1,048,576 | 65,536 | Khusus dioptimasi untuk integrasi function calling dan custom tool kompleks. |
| `gemini-3.1-flash-lite` | Gemini 3.1 Flash Lite | 1,048,576 | 65,536 | Varian ringan stabil untuk respons latensi rendah. |
| `gemini-flash-latest` | Gemini Flash Latest | 1,048,576 | 65,536 | Penunjuk otomatis (*alias*) ke versi stabil terbaru dari keluarga Flash. |
| `gemini-pro-latest` | Gemini Pro Latest | 1,048,576 | 65,536 | Penunjuk otomatis (*alias*) ke versi stabil terbaru dari keluarga Pro. |

---

## 2. Pembuatan Video AI (Google Veo)
Model AI generatif tercanggih untuk mengubah teks atau gambar menjadi video berkualitas tinggi (resolusi hingga 1080p).

| Model ID | Display Name | Input Limit | Output Limit | Kegunaan Utama |
| :--- | :--- | :--- | :--- | :--- |
| `veo-3.1-generate-preview` | **Veo 3.1** | 480 token | 8,192 token | **Kualitas terbaik**. Menghasilkan video sinematik dengan pemahaman fisika nyata, gerakan halus, dan resolusi 720p/1080p (durasi 4, 6, 8 detik). |
| `veo-3.1-fast-generate-preview` | Veo 3.1 Fast | 480 token | 8,192 token | Dioptimasi untuk waktu render yang lebih cepat dibandingkan varian standar. |
| `veo-3.1-lite-generate-preview` | Veo 3.1 Lite | 480 token | 8,192 token | Versi hemat komputasi untuk pembuatan prototipe video atau kebutuhan storyboard. |

> **Catatan Pemakaian Veo**: Bekerja secara asinkron (*Long-Running Operation*). Mendukung input teks (*Text-to-Video*) maupun gambar diam (*Image-to-Video*).

---

## 3. Pembuatan & Editing Gambar (Nano Banana)
Keluarga model difusi gambar dari Google untuk membuat gambar baru dan pengeditan berbasis prompt.

| Model ID | Display Name | Input Limit | Output Limit | Kegunaan Utama |
| :--- | :--- | :--- | :--- | :--- |
| `gemini-3-pro-image` / `nano-banana-pro-preview` | **Nano Banana Pro** | 131,072 | 32,768 | Model generasi gambar resolusi tinggi dengan detail tajam, rendering teks akurat, dan gaya fotorealistik. |
| `gemini-3.1-flash-image` / `gemini-3.1-flash-image-preview` | **Nano Banana 2** | 65,536 | 65,536 | Generasi gambar berkecepatan tinggi untuk aplikasi interaktif atau thumbnail otomatis. |
| `gemini-3.1-flash-lite-image` | Nano Banana 2 Lite | 65,536 | 65,536 | Varian gambar hemat sumber daya untuk pembuatan visual cepat dalam volume besar. |
| `gemini-2.5-flash-image` | Nano Banana | 32,768 | 32,768 | Model generasi gambar generasi awal. |

---

## 4. Audio, Text-to-Speech (TTS) & Transkripsi
Model khusus untuk sintesis suara manusia alami dan transkripsi audio ke teks dengan akurasi tinggi.

| Model ID | Display Name | Input Limit | Output Limit | Kegunaan Utama |
| :--- | :--- | :--- | :--- | :--- |
| `gemini-3.8-flash-tts` | **Gemini 3.8 Flash TTS** | 8,192 | 16,384 | Sintesis suara (Text-to-Speech) generasi terbaru dengan intonasi natural, emosi, dan berbagai pilihan suara. |
| `gemini-3.8-flash-lite-tts` | Gemini 3.8 Flash Lite TTS | 8,192 | 16,384 | Versi TTS ringan dengan latensi minimum, cocok untuk asisten suara interaktif. |
| `gemini-3.1-flash-tts-preview` | Gemini 3.1 Flash TTS | 8,192 | 16,384 | Versi preview TTS generasi 3.1. |
| `gemini-3.5-transcribe` | **Gemini 3.5 Transcribe** | 98,304 | 32,768 | Transkripsi audio panjang (podcast, rapat, wawancara) menjadi teks dengan pemisahan pembicara (*diarization*). |

---

## 5. Realtime Voice & Multimodal Streaming (Live API)
Model yang dirancang untuk komunikasi dua arah (*bidirectional streaming*) secara langsung via WebSockets / WebRTC dengan latensi milidetik (mirip berbicara dengan manusia nyata).

| Model ID | Display Name | Input Limit | Output Limit | Kegunaan Utama |
| :--- | :--- | :--- | :--- | :--- |
| `gemini-3.8-live` | **Gemini 3.8 Live** | 131,072 | 65,536 | Komunikasi suara dua arah secara langsung dengan jeda respon sangat minim dan suara alami. |
| `gemini-3.8-live-extended-thinking` | Gemini 3.8 Live Extended Think | 131,072 | 65,536 | Mode live conversation yang dilengkapi kemampuan "berpikir" (*reasoning step*) sebelum memberikan jawaban mendalam. |
| `gemini-3.5-live-translate-preview` | Gemini 3.5 Live Translate | 16,384 | 32,768 | Penerjemah suara instan real-time antar bahasa (speech-to-speech interpreter). |
| `gemini-3.5-transcribe-live` | Gemini 3.5 Transcribe Live | 131,072 | 65,536 | Transkripsi audio siaran langsung (*live captioning*) secara real-time. |
| `gemini-2.5-flash-native-audio-latest` | Gemini 2.5 Flash Native Audio | 131,072 | 8,192 | Streaming pemrosesan audio murni tanpa konversi teks perantara. |

---

## 6. Generasi Musik & Audio Efek (Lyria)
Teknologi AI dari Google DeepMind yang didedikasikan untuk penciptaan musik, instrumen, dan lanskap suara (*soundscape*).

| Model ID | Display Name | Input Limit | Output Limit | Kegunaan Utama |
| :--- | :--- | :--- | :--- | :--- |
| `lyria-realtime-exp` | **Lyria Realtime Experimental** | 1,048,576 | 65,536 | Menghasilkan musik secara real-time dan interaktif (`bidiGenerateMusic`). |
| `lyria-3.5` | Lyria 3.5 | 1,048,576 | 65,536 | Komposisi lagu dan audio soundtrack lengkap berkualitas studio dari prompt deskripsi. |
| `lyria-3-pro-preview` | Lyria 3 Pro Preview | 1,048,576 | 65,536 | Model pembuatan musik profesional dengan kontrol gaya, instrumen, dan tempo. |
| `lyria-3-clip-preview` | Lyria 3 Clip Preview | 1,048,576 | 65,536 | Pembuatan cuplikan audio pendek (jingle, ringtone, sound effect). |

---

## 7. AI Agent, Coding & Computer Use
Model yang dilatih secara khusus untuk bertindak sebagai agen otonom: menulis kode, menjalankan debugging, menggunakan komputer (mouse/keyboard), hingga interaksi robotika.

| Model ID | Display Name | Input Limit | Output Limit | Kegunaan Utama |
| :--- | :--- | :--- | :--- | :--- |
| `antigravity-preview-latest` | **Antigravity Agent Preview** | 1,048,576 | 65,536 | Model agen otonom untuk rekayasa perangkat lunak (coding, terminal command, refactoring, planning arsitektur kode). |
| `gemini-2.5-computer-use-preview-10-2025` | **Gemini Computer Use** | 131,072 | 65,536 | Mengoperasikan GUI komputer: menganalisis screenshot layar, menggerakkan kursor, klik, dan mengetik keyboard secara otomatis. |
| `gemini-robotics-er-2-preview` | Gemini Robotics-ER 2 | 131,072 | 65,536 | *Embodied Robotics*: Pemahaman spasial 3D, pengenalan objek fisik, dan perencanaan manipulasi robotik di dunia nyata. |
| `gemini-robotics-er-2-streaming-preview` | Gemini Robotics-ER 2 Streaming | 131,072 | 65,536 | Versi streaming berlatensi rendah untuk kontrol kontroler robot secara langsung. |

---

## 8. Deep Research & Fakta Teratribusi
Model untuk riset ilmiah/bisnis yang kompleks, melakukan penelusuran multi-tahap, dan memberikan sitasi faktual.

| Model ID | Display Name | Input Limit | Output Limit | Kegunaan Utama |
| :--- | :--- | :--- | :--- | :--- |
| `deep-research-max-preview-04-2026` | **Deep Research Max** | 131,072 | 65,536 | Riset komprehensif mendalam: menelusuri puluhan referensi web, mensintesis data, dan menghasilkan laporan riset ilmiah berpuluh-puluh halaman secara otomatis. |
| `deep-research-preview-04-2026` | Deep Research Preview | 131,072 | 65,536 | Versi standar untuk riset pasar dan laporan investigasi topik tertentu. |
| `aqa` | Attributed Question Answering | 7,168 | 1,024 | Menjawab pertanyaan hanya berdasarkan dokumen sumber terverifikasi dengan menyertakan sitasi persis (*grounding*) guna mencegah halusinasi. |

---

## 9. Embeddings & Vector Search (RAG)
Mengubah teks dan dokumen menjadi vektor numerik untuk pencarian semantik (*semantic search*), sistem rekomendasi, dan database RAG (Retrieval-Augmented Generation).

| Model ID | Display Name | Input Limit | Dimensi / Output | Kegunaan Utama |
| :--- | :--- | :--- | :--- | :--- |
| `gemini-embedding-2` | **Gemini Embedding 2** | 8,192 token | Vektor 768/1536/3072 | **Paling direkomendasikan** untuk RAG modern. Mendukung konteks 8K token, pemahaman semantik multimodal (teks dan gambar). |
| `gemini-embedding-2-preview` | Gemini Embedding 2 Preview | 8,192 token | Vektor Multimodal | Versi preview dari embedding generasi kedua. |
| `gemini-embedding-001` | Gemini Embedding 001 | 2,048 token | Vektor 768 | Model embedding teks klasik untuk pencarian dokumen dasar. |

---

## 10. Open Weights LLM (Gemma)
Varian dari model sumber terbuka (*open weights*) Gemma buatan Google DeepMind yang di-host langsung di infrastruktur Google AI.

| Model ID | Display Name | Input Limit | Output Limit | Kegunaan Utama |
| :--- | :--- | :--- | :--- | :--- |
| `gemma-4-31b-it` | **Gemma 4 31B IT** | 262,144 | 32,768 | Model open-weights instruksi 31 miliar parameter dengan performa penalaran tinggi dan context window 256K token. |
| `gemma-4-26b-a4b-it` | Gemma 4 26B A4B IT | 262,144 | 32,768 | Model instruksi 26B yang dioptimasi untuk efisiensi inferensi tinggi. |

---

## 11. Rangkuman Rekomendasi Pemilihan Model

| Kebutuhan Anda | Rekomendasi Model | Alasan |
| :--- | :--- | :--- |
| **Aplikasi Chat & Coding Umum** | `gemini-3.8-flash` | Sangat cepat, pintar, hemat biaya, 1M token context. |
| **Logika Rumit & Solusi Arsitektur** | `gemini-3.1-pro-preview` | Tingkat penalaran mendalam untuk masalah kompleks. |
| **Membuat Video dari Teks/Gambar** | `veo-3.1-generate-preview` | Kualitas sinematik 1080p, gerakan mulus, stabil. |
| **Membuat Gambar / Ilustrasi** | `gemini-3-pro-image` | Detail fotorealistik tajam dan render teks rapi. |
| **Mengubah Teks ke Suara (TTS)** | `gemini-3.8-flash-tts` | Intonasi manusiawi dan respons latensi rendah. |
| **Percakapan Suara Realtime (Live)** | `gemini-3.8-live` | Streaming dua arah tanpa jeda konversi. |
| **Riset Otomatis & Laporan Panjang**| `deep-research-max-preview-04-2026` | Melakukan web browsing dan sintesis ratusan sumber mandiri. |
| **Sistem RAG / Semantic Search** | `gemini-embedding-2` | Embedding multimodal 8K token terbaik dari Google. |
