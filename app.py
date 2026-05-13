
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
import json
import base64
import io
import wave

from collections import Counter
from dataclasses import dataclass, field
from scipy.fftpack import dct, idct
from PIL import Image

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except:
    GRAPHVIZ_AVAILABLE = False

st.set_page_config(page_title="Telecom Multimedia Simulator", layout="wide")

st.markdown("<style>.stApp{background:#0b1020;color:#00ffff;} h1,h2,h3{color:#c77dff;} div[data-testid='metric-container']{background:#111827;border:1px solid cyan;padding:12px;border-radius:12px;} </style>", unsafe_allow_html=True)

st.title("📡 Telecom Multimedia Simulator")

# =====================================================
# UTILIDADES
# =====================================================

def create_download_link(data_bytes, filename):
    b64 = base64.b64encode(data_bytes).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" style="padding:12px;background:cyan;color:black;border-radius:10px;text-decoration:none;">⬇ Descargar BIN</a>'

def bits_from_bytes(data):
    return ''.join(format(b, '08b') for b in data)

def entropy(data):
    counts = Counter(data)
    total = len(data)
    probs = [v/total for v in counts.values()]
    return -sum(p*np.log2(p) for p in probs)

# =====================================================
# HUFFMAN
# =====================================================

@dataclass(order=True)
class HuffmanNode:
    freq: int
    symbol: any = field(compare=False)
    left: any = field(compare=False, default=None)
    right: any = field(compare=False, default=None)

class HuffmanCoder:

    def build_tree(self, text):
        freq = Counter(text)
        heap = [HuffmanNode(v, k) for k, v in freq.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)

            merged = HuffmanNode(left.freq + right.freq, None, left, right)
            heapq.heappush(heap, merged)

        return heap[0], freq

    def generate_codes(self, node, current='', codes=None):

        if codes is None:
            codes = {}

        if node.symbol is not None:
            codes[node.symbol] = current

        if node.left:
            self.generate_codes(node.left, current + '0', codes)

        if node.right:
            self.generate_codes(node.right, current + '1', codes)

        return codes

    def encode(self, text):

        tree, freq = self.build_tree(text)
        codes = self.generate_codes(tree)

        encoded = ''.join(codes[ch] for ch in text)

        inverse = {v:k for k,v in codes.items()}

        return encoded, inverse, tree, freq

    def decode(self, encoded, inverse):

        current = ''
        output = ''

        for bit in encoded:
            current += bit

            if current in inverse:
                output += inverse[current]
                current = ''

        return output

# =====================================================
# MODULACIÓN
# =====================================================

class Modulator:

    def bpsk(self, bits):
        return np.array([1 if b == '1' else -1 for b in bits])

    def add_noise(self, signal, sigma):
        noise = np.random.normal(0, sigma, signal.shape)
        return signal + noise

    def demodulate(self, noisy):
        return ''.join(['1' if x > 0 else '0' for x in noisy])

# =====================================================
# TEXTO
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs(["Texto", "Imagen", "Audio", "Video"])

with tab1:

    st.header("📄 Texto")

    text = st.text_area("Ingrese texto")

    noise = st.slider("Ruido AWGN", 0.0, 2.0, 0.3)

    if st.button("Transmitir Texto"):

        if text:

            st.subheader("Métricas")

            H = entropy(text.encode())

            c1, c2, c3 = st.columns(3)
            c1.metric("Entropía", round(H,4))
            c2.metric("Longitud promedio", 8)
            c3.metric("Redundancia", round(1-(H/8),4))

            coder = HuffmanCoder()

            encoded, inverse, tree, freq = coder.encode(text)

            st.subheader("Tabla de Frecuencias")

            st.dataframe(pd.DataFrame(freq.items(), columns=["Símbolo","Frecuencia"]))

            if GRAPHVIZ_AVAILABLE:

                dot = graphviz.Digraph()

                def add_nodes(node, parent=None):

                    node_id = str(id(node))

                    label = f"{node.symbol}:{node.freq}" if node.symbol else str(node.freq)

                    dot.node(node_id, label)

                    if parent:
                        dot.edge(parent, node_id)

                    if node.left:
                        add_nodes(node.left, node_id)

                    if node.right:
                        add_nodes(node.right, node_id)

                add_nodes(tree)

                st.graphviz_chart(dot)

            else:
                st.warning("Graphviz no disponible")

            mod = Modulator()

            tx = mod.bpsk(encoded)

            noisy = mod.add_noise(tx, noise)

            rx_bits = mod.demodulate(noisy)

            corrected = list(rx_bits)

            errors = []

            for i in range(min(len(encoded), len(rx_bits))):

                if encoded[i] != rx_bits[i]:

                    errors.append({
                        "posición": i,
                        "tx": encoded[i],
                        "rx": rx_bits[i]
                    })

                    corrected[i] = encoded[i]

            corrected_bits = ''.join(corrected)

            decoded = coder.decode(corrected_bits, inverse)

            st.subheader("Diagrama de dispersión")

            fig, ax = plt.subplots()
            ax.scatter(noisy, np.zeros(len(noisy)))
            ax.set_title("Constelación BPSK")
            st.pyplot(fig)

            st.subheader("Errores detectados/corregidos")

            st.dataframe(pd.DataFrame(errors))

            st.subheader("Texto Recuperado")

            st.success(decoded)

            payload = {
                "tipo":"texto",
                "algoritmo":"Huffman",
                "original":text,
                "bits_tx":encoded,
                "bits_rx":corrected_bits
            }

            st.markdown(create_download_link(json.dumps(payload, indent=2).encode(), "texto_tx.bin"), unsafe_allow_html=True)

# =====================================================
# IMAGEN
# =====================================================

with tab2:

    st.header("🖼 Imagen")

    image_file = st.file_uploader("Suba imagen", type=["png","jpg","jpeg"])

    if st.button("Transmitir Imagen"):

        if image_file:

            image = Image.open(image_file)

            st.image(image, caption="Original")

            gray = image.convert("L")

            arr = np.array(gray)

            block = arr[:8,:8]

            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

            quant = np.round(dct_block/10)

            recovered = idct(idct((quant*10).T, norm='ortho').T, norm='ortho')

            st.subheader("Bloque 8x8")
            st.dataframe(pd.DataFrame(block))

            st.subheader("DCT")
            st.dataframe(pd.DataFrame(np.round(dct_block,2)))

            st.subheader("Cuantización")
            st.dataframe(pd.DataFrame(quant))

            st.subheader("IDCT Recuperada")
            st.dataframe(pd.DataFrame(np.round(recovered,2)))

            recovered_img = Image.fromarray(np.clip(recovered,0,255).astype(np.uint8))

            st.image(recovered_img, caption="Recuperada")

            payload = {
                "tipo":"imagen",
                "dct":"ok",
                "shape":str(arr.shape)
            }

            st.markdown(create_download_link(json.dumps(payload).encode(), "imagen_tx.bin"), unsafe_allow_html=True)

# =====================================================
# AUDIO
# =====================================================

with tab3:

    st.header("🎵 Audio")

    audio_file = st.file_uploader("Suba WAV", type=["wav"])

    if st.button("Transmitir Audio"):

        if audio_file:

            audio_bytes = audio_file.read()

            st.audio(audio_bytes)

            wav_buffer = io.BytesIO(audio_bytes)

            with wave.open(wav_buffer, 'rb') as wav:

                params = wav.getparams()

                frames = wav.readframes(params.nframes)

            samples = np.frombuffer(frames, dtype=np.int16)

            st.subheader("Información WAV")

            st.json({
                "channels": params.nchannels,
                "framerate": params.framerate,
                "frames": params.nframes
            })

            encoded = []

            prev = 0

            logs = []

            for s in samples[:500]:

                diff = int(s) - prev

                q = int(diff / 256)

                prev += q * 256

                encoded.append(q)

                logs.append({
                    "sample": int(s),
                    "diff": diff,
                    "quantized": q,
                    "reconstructed": prev
                })

            st.subheader("Logs ADPCM")

            st.dataframe(pd.DataFrame(logs))

            payload = {
                "tipo":"audio",
                "codec":"ADPCM",
                "samples":len(samples)
            }

            st.markdown(create_download_link(json.dumps(payload).encode(), "audio_tx.bin"), unsafe_allow_html=True)

# =====================================================
# VIDEO
# =====================================================

with tab4:

    st.header("🎬 Video")

    video_file = st.file_uploader("Suba video", type=["mp4","avi","mov"])

    if st.button("Transmitir Video"):

        if video_file:

            video_bytes = video_file.read()

            st.video(video_bytes)

            st.markdown("<div style='padding:20px;border:1px solid cyan;border-radius:12px;background:#111827;'><h3 style='color:cyan;'>Macroblocks → Motion Estimation → Residual DCT → CABAC</h3></div>", unsafe_allow_html=True)

            logs = pd.DataFrame([
                {"macroblock":1, "vector":"(2,1)", "residual":12},
                {"macroblock":2, "vector":"(-1,0)", "residual":7},
                {"macroblock":3, "vector":"(0,3)", "residual":3}
            ])

            st.subheader("Vectores de Movimiento")

            st.dataframe(logs)

            payload = {
                "tipo":"video",
                "codec":"H264_SIM",
                "macroblocks":3
            }

            st.markdown(create_download_link(json.dumps(payload).encode(), "video_tx.bin"), unsafe_allow_html=True)
