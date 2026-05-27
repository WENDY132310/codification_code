
from __future__ import annotations

import base64
import heapq
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="Multimedia Telecom Lab",
    page_icon="📡",
    layout="wide"
)

# =============================================================================
# CSS
# =============================================================================

st.markdown("""
<style>

.stApp{
    background:#050816;
    color:white;
}

h1,h2,h3{
    color:#00ffff;
}

div[data-testid='metric-container']{
    background:#111827;
    border:1px solid #7c3aed;
    border-radius:14px;
    padding:14px;
}

.stButton button{
    background:linear-gradient(135deg,#06b6d4,#7c3aed);
    color:white;
    border:none;
    border-radius:10px;
    font-weight:bold;
}

.matrix-cell {
    display:inline-block;
    width:32px;
    height:32px;
    line-height:32px;
    text-align:center;
    border:1px solid #333;
    margin:2px;
    font-family:monospace;
    border-radius:6px;
}

.data-bit {
    background:#1d4ed8;
}

.parity-bit {
    background:#065f46;
    font-weight:bold;
}

.error-bit {
    background:#991b1b;
    color:white;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPERS
# =============================================================================

def bytes_to_bits(data):
    return ''.join(format(b, '08b') for b in data)

def inject_bit_errors(bits, ber):

    bit_list = list(bits)

    total_errors = int(len(bit_list) * (ber / 2.0))

    for _ in range(total_errors):

        idx = random.randint(0, len(bit_list)-1)

        bit_list[idx] = (
            '1'
            if bit_list[idx] == '0'
            else '0'
        )

    return ''.join(bit_list)

def create_download_link(payload, filename):

    payload_json = json.dumps(payload)

    b64 = base64.b64encode(
        payload_json.encode()
    ).decode()

    href = f'''
    <a href="data:application/octet-stream;base64,{b64}"
    download="{filename}">
    ⬇ Descargar {filename}
    </a>
    '''

    return href

# =============================================================================
# MÉTRICAS
# =============================================================================

@dataclass
class EstadisticasInfo:

    entropia: float
    longitud_promedio: float
    eficiencia: float
    redundancia: float

class AnalizadorInformacion:

    def __init__(self, symbols):

        self.symbols = symbols

    def calcular(self):

        freq = Counter(self.symbols)

        total = sum(freq.values())

        probs = {
            k:v/total
            for k,v in freq.items()
        }

        H = -sum(
            p * math.log2(p)
            for p in probs.values()
            if p > 0
        )

        L = sum(
            p * math.ceil(-math.log2(p))
            for p in probs.values()
        )

        efficiency = H / L if L > 0 else 0

        redundancy = 1 - efficiency

        return EstadisticasInfo(
            entropia=H,
            longitud_promedio=L,
            eficiencia=efficiency,
            redundancia=redundancy
        )

# =============================================================================
# HUFFMAN
# =============================================================================

@dataclass(order=True)
class HuffNode:

    freq: int
    symbol: any = field(compare=False)

    left: any = field(compare=False, default=None)
    right: any = field(compare=False, default=None)

class HuffmanCoder:

    def encode(self, text):

        freq = Counter(text)

        heap = [
            HuffNode(v, k)
            for k,v in freq.items()
        ]

        heapq.heapify(heap)

        while len(heap) > 1:

            l = heapq.heappop(heap)
            r = heapq.heappop(heap)

            heapq.heappush(
                heap,
                HuffNode(
                    l.freq + r.freq,
                    None,
                    l,
                    r
                )
            )

        codes = {}

        def generate(node, current=''):

            if node.symbol is not None:
                codes[node.symbol] = current or '0'

            if node.left:
                generate(node.left, current+'0')

            if node.right:
                generate(node.right, current+'1')

        if heap:
            generate(heap[0])

        encoded = ''.join(
            codes[ch]
            for ch in text
        )

        inverse = {
            v:k for k,v in codes.items()
        }

        return encoded, inverse, codes

# =============================================================================
# RLE
# =============================================================================

class RLECoder:

    def encode(self, text):

        if not text:
            return ''

        out = []

        count = 1
        prev = text[0]

        for ch in text[1:]:

            if ch == prev:
                count += 1

            else:
                out.append(f"{count}:{prev}")
                prev = ch
                count = 1

        out.append(f"{count}:{prev}")

        return '|'.join(out)

# =============================================================================
# MODULACIÓN
# =============================================================================

class Modulator:

    def __init__(self, scheme):

        self.scheme = scheme

    def modulate(self, bits):

        if self.scheme == 'BPSK':

            return np.array([
                1 if b == '1' else -1
                for b in bits
            ])

        else:

            return np.array([
                (
                    1 if bits[i] == '1' else -1,
                    1 if bits[i+1] == '1' else -1
                )
                for i in range(0, len(bits)-1, 2)
            ])

    def channel_awgn(self, signal, ber):

        return signal + np.random.normal(
            0,
            ber * 2.0,
            signal.shape
        )

    def render_constellation(self, noisy_signal):

        fig, ax = plt.subplots(figsize=(5,4))

        fig.patch.set_facecolor('#050816')
        ax.set_facecolor('#111827')

        if self.scheme == 'BPSK':

            ax.scatter(
                noisy_signal,
                np.zeros(len(noisy_signal)),
                s=10
            )

        else:

            ax.scatter(
                noisy_signal[:,0],
                noisy_signal[:,1],
                s=10
            )

        ax.set_title('Constelación')

        return fig

# =============================================================================
# FEC 2D
# =============================================================================

class MatrixFEC:

    def __init__(self, cols=4):

        self.cols = cols

    def parity(self, bits):

        return (
            '1'
            if bits.count('1') % 2 != 0
            else '0'
        )

    def encode(self, bits):

        padding = (
            self.cols - (len(bits)%self.cols)
        ) % self.cols

        bits += '0' * padding

        rows = len(bits)//self.cols

        encoded = ''

        html = '<h4>Paso 1 — Matriz Original</h4><div>'

        col_parity = ['0'] * self.cols

        for r in range(rows):

            row = bits[
                r*self.cols:(r+1)*self.cols
            ]

            for b in row:

                html += f'<span class="matrix-cell data-bit">{b}</span>'

            html += '<br>'

            p = self.parity(row)

            for i,b in enumerate(row):

                if b == '1':

                    col_parity[i] = (
                        '0'
                        if col_parity[i]=='1'
                        else '1'
                    )

            encoded += row + p

        html += '</div>'

        return encoded, padding, html

    def syndrome_demo(self, encoded_bits):

        bits = list(encoded_bits)

        error_index = random.randint(0, len(bits)-1)

        bits[error_index] = (
            '1'
            if bits[error_index] == '0'
            else '0'
        )

        html = '<h4>Paso 3 — Error Inyectado</h4><div>'

        for i,b in enumerate(bits):

            css = (
                'error-bit'
                if i == error_index
                else 'data-bit'
            )

            html += f'<span class="matrix-cell {css}">{b}</span>'

        html += '</div>'

        return ''.join(bits), html, error_index

# =============================================================================
# HEADER
# =============================================================================

st.title("📡 Multimedia Telecom Lab")

st.markdown(
    '''
<div style="background:#0f172a;padding:16px;border-radius:12px;">
<h3>Fuente + Canal + FEC + AWGN + Modulación + Reconstrucción</h3>
</div>
''',
    unsafe_allow_html=True
)

# =============================================================================
# MAIN TABS
# =============================================================================

main_tabs = st.tabs([
    "📄 Texto",
    "🖼 Imagen",
    "🎵 Audio",
    "🎬 Video"
])

# =============================================================================
# TEMPLATE MULTIMEDIA
# =============================================================================

def multimedia_pipeline(title, payload_text):

    subtabs = st.tabs([
        "Fuente",
        "Canal",
        "Pipeline Completo"
    ])

    # =========================================================================
    # FUENTE
    # =========================================================================

    with subtabs[0]:

        st.header(f"Fuente — {title}")

        text = st.text_area(
            "Payload",
            payload_text,
            key=f"{title}_payload"
        )

        algo = st.selectbox(
            "Algoritmo",
            ["Huffman", "RLE"],
            key=f"{title}_algo"
        )

        if st.button(
            f"Codificar Fuente {title}"
        ):

            stats = AnalizadorInformacion(
                list(text)
            ).calcular()

            c1,c2,c3,c4 = st.columns(4)

            with c1:
                st.metric(
                    "Entropía",
                    f"{stats.entropia:.4f}"
                )

            with c2:
                st.metric(
                    "Longitud Promedio",
                    f"{stats.longitud_promedio:.4f}"
                )

            with c3:
                st.metric(
                    "Eficiencia",
                    f"{stats.eficiencia*100:.2f}%"
                )

            with c4:
                st.metric(
                    "Redundancia",
                    f"{stats.redundancia*100:.2f}%"
                )

            if algo == 'Huffman':

                encoded, inverse, codes = (
                    HuffmanCoder().encode(text)
                )

                st.code(encoded[:1000])

                st.json(codes)

            else:

                encoded = (
                    RLECoder().encode(text)
                )

                st.code(encoded)

    # =========================================================================
    # CANAL
    # =========================================================================

    with subtabs[1]:

        st.header(f"Canal — {title}")

        bits = st.text_area(
            "Bits",
            "101010111100001111",
            key=f"{title}_bits"
        )

        modulation = st.selectbox(
            "Modulación",
            ["BPSK", "QPSK"],
            key=f"{title}_mod"
        )

        fec_cols = st.selectbox(
            "FEC",
            [4,8,16],
            key=f"{title}_fec"
        )

        ber = st.slider(
            "BER",
            0.0,
            1.0,
            0.05,
            key=f"{title}_ber"
        )

        if st.button(
            f"Transmitir {title}"
        ):

            fec = MatrixFEC(fec_cols)

            encoded_fec, padding, html = (
                fec.encode(bits)
            )

            st.markdown(
                html,
                unsafe_allow_html=True
            )

            corrupted, html_error, idx = (
                fec.syndrome_demo(
                    encoded_fec
                )
            )

            st.markdown(
                html_error,
                unsafe_allow_html=True
            )

            st.code(
                f'''
Paso 4 — Síndrome

Error detectado en índice:
{idx}

Paso 5 — Corrección automática
'''
            )

            mod = Modulator(modulation)

            signal = mod.modulate(corrupted)

            noisy = mod.channel_awgn(
                signal,
                ber
            )

            st.pyplot(
                mod.render_constellation(
                    noisy
                )
            )

            st.metric(
                "Bits Transmitidos",
                len(bits)
            )

            st.metric(
                "Bits Canal",
                len(encoded_fec)
            )

    # =========================================================================
    # PIPELINE
    # =========================================================================

    with subtabs[2]:

        st.header(f"Pipeline Completo — {title}")

        payload = st.text_area(
            "Payload Pipeline",
            payload_text,
            key=f"{title}_pipe_payload"
        )

        source_algorithm = st.selectbox(
            "Fuente",
            ["Huffman", "RLE"],
            key=f"{title}_pipe_source"
        )

        modulation = st.selectbox(
            "Modulación",
            ["BPSK", "QPSK"],
            key=f"{title}_pipe_mod"
        )

        ber = st.slider(
            "BER AWGN",
            0.0,
            1.0,
            0.02,
            key=f"{title}_pipe_ber"
        )

        if st.button(
            f"Ejecutar Pipeline {title}"
        ):

            if source_algorithm == 'Huffman':

                encoded, inverse, codes = (
                    HuffmanCoder().encode(
                        payload
                    )
                )

                source_bits = encoded

            else:

                encoded = (
                    RLECoder().encode(
                        payload
                    )
                )

                source_bits = bytes_to_bits(
                    encoded.encode()
                )

            st.subheader("1️⃣ Fuente")

            st.code(source_bits[:1000])

            fec = MatrixFEC(4)

            fec_bits, padding, html = (
                fec.encode(source_bits)
            )

            st.subheader("2️⃣ FEC")

            st.markdown(
                html,
                unsafe_allow_html=True
            )

            mod = Modulator(modulation)

            signal = mod.modulate(fec_bits)

            noisy = mod.channel_awgn(
                signal,
                ber
            )

            st.subheader("3️⃣ Canal AWGN")

            st.pyplot(
                mod.render_constellation(
                    noisy
                )
            )

            rx_bits = inject_bit_errors(
                fec_bits,
                ber
            )

            st.subheader("4️⃣ Rx")

            st.code(rx_bits[:1000])

            payload_export = {

                "title": title,
                "fuente": source_algorithm,
                "modulacion": modulation,
                "ber": ber,
                "payload_original": payload,
                "tx_bits": fec_bits,
                "rx_bits": rx_bits
            }

            st.markdown(
                create_download_link(
                    payload_export,
                    f"{title}_pipeline.bin"
                ),
                unsafe_allow_html=True
            )

            st.success(
                "Pipeline ejecutado correctamente"
            )

# =============================================================================
# TEXTO
# =============================================================================

with main_tabs[0]:

    multimedia_pipeline(
        "Texto",
        "HOLA TELECOMUNICACIONES"
    )

# =============================================================================
# IMAGEN
# =============================================================================

with main_tabs[1]:

    st.header("🖼 Imagen")

    image_file = st.file_uploader(
        "Sube Imagen",
        type=["png","jpg","jpeg"]
    )

    if image_file:

        img = Image.open(image_file)

        st.image(img)

    multimedia_pipeline(
        "Imagen",
        "PIXELS_COMPRESSED"
    )

# =============================================================================
# AUDIO
# =============================================================================

with main_tabs[2]:

    st.header("🎵 Audio")

    audio_file = st.file_uploader(
        "Sube WAV",
        type=["wav"]
    )

    if audio_file:

        st.audio(audio_file)

    multimedia_pipeline(
        "Audio",
        "PCM_AUDIO_STREAM"
    )

# =============================================================================
# VIDEO
# =============================================================================

with main_tabs[3]:

    st.header("🎬 Video")

    video_file = st.file_uploader(
        "Sube Video",
        type=["mp4"]
    )

    if video_file:

        st.video(video_file)

    multimedia_pipeline(
        "Video",
        "VIDEO_FRAME_STREAM"
    )
