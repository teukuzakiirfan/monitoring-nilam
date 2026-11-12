import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model_nilam_cnn.keras"
IMG_SIZE = (128, 128)
# Urutan harus sama dengan class_names di notebook (alfabet)
CLASS_NAMES = ["nilam budok", "nilam layu", "nilam sehat"]
THRESHOLD = 0.60  # di bawah ini hasil dianggap kurang yakin

INFO = {
    "nilam budok": {
        "judul": "Terindikasi Penyakit Budok",
        "level": "error",
        "penyebab": (
            "Jamur *Synchytrium pogostemonis* (parasit obligat, tular tanah). "
            "Menyerang daun, tangkai daun, dan batang, terutama bagian pucuk. "
            "Dikenal juga sebagai penyakit **karat palsu** karena gejalanya mirip karat tetapi tidak berspora."
        ),
        "gejala": [
            "Bintik/bercak mirip karat pada daun, tangkai, dan batang",
            "Pucuk dan daun muda terganggu, pertumbuhan tanaman terhambat",
            "Pada serangan berat produksi terna dan minyak nilam turun",
        ],
        "penanggulangan": [
            "**Sanitasi:** pangkas atau cabut bagian/tanaman yang bergejala, lalu musnahkan (bakar/kubur), jangan dijadikan bahan setek.",
            "**Benih sehat:** gunakan setek dari tanaman yang bebas gejala agar penyakit tidak menyebar ke lahan lain.",
            "**Hayati/nabati:** agensia hayati (rizobakteri, *Trichoderma*) atau pestisida nabati (campuran minyak mimba dan serai wangi ±0,5%).",
            "**Kimia (hanya jika serangan berat):** fungisida seperti benomil atau tembaga oksida sesuai dosis anjuran dan izin edar yang berlaku.",
            "**Budidaya:** perbaiki drainase dan jaga kebersihan lahan.",
        ],
        "langkah": [
            ("Segera (hari ini)", "Isolasi tanaman bergejala dan hentikan pengambilan setek dari tanaman tersebut."),
            ("Minggu ini", "Pangkas dan musnahkan bagian sakit, cek tanaman di sekitarnya, lalu terapkan pengendalian hayati/nabati."),
            ("Jangka panjang", "Pakai benih bersertifikat/sehat, pantau lahan rutin tiap minggu, dan jangan memindahkan tanah atau bibit dari lahan terinfeksi."),
        ],
    },
    "nilam layu": {
        "judul": "Terindikasi Penyakit Layu",
        "level": "warning",
        "penyebab": (
            "Paling umum **layu bakteri** oleh *Ralstonia solanacearum*, yang masuk lewat luka akar atau alat pertanian. "
            "Serangan **nematoda** (puru akar) sering memperparah dan membuka jalan bagi bakteri. "
            "Kekurangan air atau drainase buruk juga bisa menimbulkan gejala serupa, jadi perlu pengecekan akar dan batang."
        ),
        "gejala": [
            "Daun layu, awalnya pada beberapa cabang tanpa urutan teratur lalu menyebar ke seluruh tanaman",
            "Daun tampak segar kembali sore hari pada tahap awal, lalu layu permanen dan tanaman mati",
            "Batang yang dipotong dan direndam air mengeluarkan lendir putih keruh (indikasi layu bakteri)",
            "Akar berbenjol atau terkelupas (indikasi nematoda)",
        ],
        "penanggulangan": [
            "**Sanitasi:** cabut tanaman sakit beserta akarnya dan musnahkan; jangan dibuang ke saluran air kebun.",
            "**Alat bersih:** sterilkan alat pangkas/cangkul setelah dipakai pada tanaman sakit agar bakteri tidak menular lewat luka.",
            "**Drainase:** hindari genangan air dan perbaiki saluran, karena penyebaran lebih cepat saat basah.",
            "**Kendalikan nematoda:** pengolahan tanah yang baik dan penambahan bahan organik, agar luka akar berkurang.",
            "**Hayati:** pseudomonad fluoresen (*Pseudomonas fluorescens*) dapat dipakai sebagai agen antagonis.",
            "**Rotasi tanaman:** jangan menanam inang lain seperti cabai, tomat, terung, atau jahe di lahan bekas serangan.",
        ],
        "langkah": [
            ("Segera (hari ini)", "Cek batang dan akar. Jika ada lendir putih atau akar berpuru, cabut tanaman dan musnahkan."),
            ("Minggu ini", "Periksa tanaman sekitar, perbaiki drainase, dan sterilkan alat kerja."),
            ("Jangka panjang", "Gunakan bibit sehat, terapkan agen hayati, lakukan rotasi tanaman, dan hindari melukai akar saat perawatan."),
        ],
    },
    "nilam sehat": {
        "judul": "Tanaman Nilam Sehat",
        "level": "success",
        "penyebab": "Tidak terdeteksi tanda penyakit budok maupun layu pada foto ini.",
        "gejala": [
            "Daun hijau segar, tidak ada bercak atau bintik karat",
            "Tidak ada tanda layu pada daun maupun batang",
        ],
        "penanggulangan": [
            "**Pantau rutin:** periksa daun, pucuk, dan batang minimal seminggu sekali.",
            "**Sanitasi lahan:** bersihkan gulma dan sisa tanaman agar tidak jadi sumber penyakit.",
            "**Drainase baik:** hindari genangan air di sekitar akar.",
            "**Pemupukan berimbang** agar tanaman tahan terhadap serangan penyakit.",
        ],
        "langkah": [
            ("Rutin", "Ulangi pengecekan berkala dan foto ulang jika muncul bercak atau daun mulai layu."),
            ("Pencegahan", "Gunakan setek dari tanaman sehat dan hindari membawa tanah/bibit dari lahan yang pernah terserang."),
        ],
    },
}

st.set_page_config(page_title="Diagnosis Kesehatan Nilam Aceh", page_icon="🌿", layout="centered")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

st.title("🌿 Diagnosis Kesehatan Tanaman Nilam Aceh")
st.caption("Klasifikasi citra dengan Convolutional Neural Network (CNN): sehat, layu, atau budok.")

with st.sidebar:
    st.header("Tentang")
    st.write(
        "Aplikasi ini mengklasifikasikan kondisi tanaman nilam dari foto, "
        "lalu menampilkan penyebab, cara penanggulangan, dan langkah yang disarankan."
    )
    st.write("**Tips foto:** ambil dekat, cahaya cukup, fokus pada daun atau batang yang bergejala.")
    st.warning(
        "Hasil bersifat indikatif. Untuk keputusan penanganan (terutama penggunaan fungisida), "
        "konsultasikan dengan penyuluh pertanian atau BPTP setempat."
    )

tab_foto, tab_monitor = st.tabs(["📷 Diagnosis Foto", "📋 Checklist Monitoring"])

with tab_foto:
    file = st.file_uploader("Upload atau ambil foto tanaman/daun nilam", type=["jpg", "jpeg", "png", "webp"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Gambar yang dianalisis", use_container_width=True)

        arr = np.expand_dims(np.array(img.resize(IMG_SIZE), dtype="float32"), 0)
        probs = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        conf = float(probs[idx])
        info = INFO[label]
        st.session_state['hasil_foto'] = (label, conf)

        st.divider()

        if conf < THRESHOLD:
            st.warning(
                f"Model kurang yakin ({conf * 100:.1f}%). Coba foto ulang dengan pencahayaan lebih baik "
                "dan fokus pada daun/batang. Perkiraan sementara di bawah ini."
            )

        msg = f"**{info['judul']}** (tingkat keyakinan {conf * 100:.1f}%)"
        getattr(st, info["level"])(msg)

        st.write("**Probabilitas tiap kelas**")
        for name, p in zip(CLASS_NAMES, probs):
            st.progress(float(p), text=f"{name}: {p * 100:.1f}%")

        tab1, tab2, tab3 = st.tabs(["🔍 Diagnosis", "🛠️ Penanggulangan", "✅ Langkah yang disarankan"])

        with tab1:
            st.subheader("Penyebab")
            st.markdown(info["penyebab"])
            st.subheader("Ciri-ciri / gejala")
            for g in info["gejala"]:
                st.markdown(f"- {g}")

        with tab2:
            for p in info["penanggulangan"]:
                st.markdown(f"- {p}")

        with tab3:
            for waktu, isi in info["langkah"]:
                st.markdown(f"**{waktu}**")
                st.markdown(f"> {isi}")

        st.caption("Hasil bersifat indikatif dan tidak menggantikan pemeriksaan langsung oleh ahli.")

with tab_monitor:
    st.subheader("Checklist Monitoring Budidaya Nilam")
    st.caption("Isi kondisi lapangan, sistem menilai apakah sudah ideal atau perlu perbaikan.")

    c1, c2 = st.columns(2)
    with c1:
        fase = st.selectbox("Fase tanaman", ["Fase awal tanam", "Fase pertumbuhan / dewasa"])
        ph = st.number_input("pH tanah", min_value=3.0, max_value=9.0, value=6.0, step=0.1)
        lembap = st.selectbox("Kondisi kelembapan tanah",
                              ["Kering / pecah-pecah", "Lembap (tidak tergenang)", "Becek / tergenang"], index=1)
    with c2:
        naungan = st.slider("Naungan (%)", 0, 100, 40)
        gejala = st.multiselect(
            "Gejala yang terlihat",
            ["Daun menguning", "Bintik-bintik / bercak pada daun", "Daun layu",
             "Layu mendadak padahal air cukup", "Lubang pada daun", "Daun menggulung",
             "Akar bengkak / berbenjol"],
        )

    with st.expander("Pasca-panen (opsional)"):
        pa = st.number_input("Kadar Patchouli Alcohol / PA minyak (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)

    if st.button("Evaluasi", type="primary"):
        hasil = []  # (parameter, nilai, status, catatan)

        # pH
        if 5.5 <= ph <= 6.5:
            hasil.append(("pH tanah", f"{ph:.1f}", "ok", "Dalam kisaran ideal (5,5 - 6,5)."))
        elif ph < 5.5:
            hasil.append(("pH tanah", f"{ph:.1f}", "warn", "Terlalu asam, penyerapan nutrisi terganggu. Pertimbangkan pengapuran sesuai anjuran."))
        else:
            hasil.append(("pH tanah", f"{ph:.1f}", "warn", "Di atas kisaran ideal. Cek ulang pengukuran dan kondisi lahan."))

        # kelembapan
        if lembap.startswith("Lembap"):
            hasil.append(("Kelembapan tanah", "Lembap", "ok", "Ideal: lembap tetapi tidak tergenang."))
        elif lembap.startswith("Kering"):
            hasil.append(("Kelembapan tanah", "Kering", "warn", "Tanah terlalu kering. Lakukan penyiraman atau mulsa untuk menjaga kelembapan."))
        else:
            hasil.append(("Kelembapan tanah", "Tergenang", "bad", "Risiko busuk akar dan penyakit layu. Perbaiki drainase segera."))

        # naungan
        if fase == "Fase awal tanam":
            if 30 <= naungan <= 50:
                hasil.append(("Naungan", f"{naungan}%", "ok", "Sesuai kisaran fase awal (30 - 50%)."))
            elif naungan < 30:
                hasil.append(("Naungan", f"{naungan}%", "warn", "Kurang, daun muda berisiko terbakar sinar matahari langsung. Tambah naungan."))
            else:
                hasil.append(("Naungan", f"{naungan}%", "warn", "Terlalu rapat, pertumbuhan bisa terhambat dan lahan jadi lembap. Kurangi naungan."))
        else:
            hasil.append(("Naungan", f"{naungan}%", "info", "Acuan 30 - 50% berlaku untuk fase awal tanam, tidak dinilai untuk fase ini."))

        # gejala
        saran = {
            "Daun menguning": "Bisa karena kekurangan nitrogen atau infeksi virus (PogoPV). Cek pemupukan; bila disertai pola mosaik/keriting, curigai virus dan cabut tanaman.",
            "Bintik-bintik / bercak pada daun": "Periksa indikasi budok atau bercak daun. Cocokkan dengan diagnosis foto.",
            "Daun layu": "Cek akar dan batang serta kelembapan tanah, bedakan layu karena kurang air dengan layu penyakit.",
            "Layu mendadak padahal air cukup": "Indikasi kuat layu bakteri. Cabut dan musnahkan tanaman, sterilkan alat.",
            "Lubang pada daun": "Indikasi hama pemakan daun. Periksa keberadaan ulat pada permukaan bawah daun.",
            "Daun menggulung": "Indikasi ulat penggulung daun. Buang gulungan daun dan pantau populasi hama.",
            "Akar bengkak / berbenjol": "Indikasi nematoda bintil akar (Meloidogyne). Perbaiki pengolahan tanah dan tambahkan bahan organik.",
        }
        if gejala:
            for g in gejala:
                lvl = "bad" if g.startswith("Layu mendadak") else "warn"
                hasil.append(("Gejala", g, lvl, saran[g]))
        else:
            hasil.append(("Gejala", "Tidak ada", "ok", "Tidak ada gejala yang dilaporkan."))

        # pasca panen
        if pa > 0:
            if pa >= 30:
                hasil.append(("Kadar PA minyak", f"{pa:.1f}%", "ok", "Memenuhi acuan minimum (sekitar 30% atau lebih)."))
            else:
                hasil.append(("Kadar PA minyak", f"{pa:.1f}%", "warn", "Di bawah acuan minimum (sekitar 30%). Cek proses panen, pengeringan, dan penyulingan."))

        # tampilkan
        ikon = {"ok": "🟢", "warn": "🟡", "bad": "🔴", "info": "ℹ️"}
        for p, v, s, cat in hasil:
            st.markdown(f"{ikon[s]} **{p}: {v}**  \n{cat}")

        n_bad = sum(1 for h in hasil if h[2] == "bad")
        n_warn = sum(1 for h in hasil if h[2] == "warn")
        st.divider()
        if n_bad:
            st.error(f"Status keseluruhan: PERLU TINDAKAN SEGERA ({n_bad} kritis, {n_warn} perlu perbaikan).")
        elif n_warn:
            st.warning(f"Status keseluruhan: PERLU PERBAIKAN ({n_warn} parameter).")
        else:
            st.success("Status keseluruhan: kondisi budidaya baik.")

        # gabung dengan diagnosis foto
        if "hasil_foto" in st.session_state:
            lbl, cf = st.session_state["hasil_foto"]
            st.info(f"Diagnosis foto terakhir: **{lbl}** ({cf * 100:.1f}%). "
                    "Pertimbangkan bersama hasil checklist di atas.")

    st.caption("Rentang acuan: pH 5,5 - 6,5, naungan 30 - 50% (fase awal), kadar PA minimal sekitar 30%. Sesuaikan dengan anjuran penyuluh/BPTP setempat.")
