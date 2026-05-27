# =============================================================================
# MULTIMEDIA TELECOM LAB — FUENTE + CANAL UNIFICADOS
# STREAMLIT APP COMPLETA
# =============================================================================

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
import json
import base64
import io
import wave
import random
import math
import struct
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import Counter
from scipy.fftpack import dct, idct
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

textarea{
    background:#0f172a !important;
    color:white !important;
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
    width:26px;
    height:26px;
    line-height:26px;
    text-align:center;
    border:1px solid #333;
    margin:1px;
    font-family:monospace;
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

def bits_to_bytes(bits):
    return bytes(
        int(bits[i:i+8], 2)
        for i in range(0, len(bits), 8)
        if i+8 <= len(bits)
    )

def inject_bit_errors(bits, ber):

    if ber <= 0:
        return bits

    bit_list = list(bits)

    for _ in range(int(len(bit_list) * (ber / 1.5))):

        idx = random.randint(0, len(bit_list)-1)

        bit_list[idx] = (
            '1' if bit_list[idx] == '0'
            else '0'
        )

    return "".join(bit_list)

def create_download_link(payload_dict, filename):

    json_str = json.dumps(payload_dict)

    b64 = base64.b64encode(
        json_str.encode()
    ).decode()

    href = f'''
    <a href="data:application/octet-stream;base64,{b64}"
    download="{filename}">
    ⬇ Descargar {filename}
    </a>
    '''

    return href

# =============================================================================
# ANALISIS DE FUENTE
# =============================================================================

@dataclass
class EstadisticasInfo:

    entropia: float
    simbolos_unicos: int
    total_simbolos: int

class AnalizadorTexto:

    def __init__(self, texto):

        self.texto = texto

    def calcular(self):

        freq = Counter(self.texto)

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

        return EstadisticasInfo(
            entropia=H,
            simbolos_unicos=len(freq),
            total_simbolos=total
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

        def gen(node, current=''):

            if node.symbol is not None:

                codes[node.symbol] = (
                    current if current else '0'
                )

            if node.left:
                gen(node.left, current+'0')

            if node.right:
                gen(node.right, current+'1')

        if heap:
            gen(heap[0])

        encoded = ''.join(
            codes[ch]
            for ch in text
        )

        inverse = {
            v:k for k,v in codes.items()
        }

        return encoded, inverse, codes

    def decode(self, bits, inverse):

        current = ''
        output = ''

        for b in bits:

            current += b

            if current in inverse:

                output += inverse[current]

                current = ''

        return output

# =============================================================================
# RLE
# =============================================================================

class RLECoder:

    def encode(self, text):

        if not text:
            return ""

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

        return "|".join(out)

    def decode(self, encoded):

        result = ""

        parts = encoded.split("|")

        for p in parts:

            c, ch = p.split(":")

            result += ch * int(c)

        return result

# =============================================================================
# MODULADOR
# =============================================================================

class Modulator:

    def __init__(self, scheme):

        self.scheme = scheme

    def modulate(self, bits):

        if self.scheme == "BPSK":

            return np.array([
                1 if b == '1' else -1
                for b in bits
            ])

        elif self.scheme == "QPSK":

            return np.array([
                {
                    '00':(-1,-1),
                    '01':(-1,1),
                    '10':(1,-1),
                    '11':(1,1)
                }[
                    bits[i:i+2].ljust(2,'0')
                ]
                for i in range(0,len(bits),2)
            ])

        else:

            levels = [-3,-1,1,3]

            return np.array([
                (
                    levels[int(bits[i:i+4].ljust(4,'0')[:2],2)],
                    levels[int(bits[i:i+4].ljust(4,'0')[2:],2)]
                )
                for i in range(0,len(bits),4)
            ])

    def channel_awgn(self, signal, ber):

        return signal + np.random.normal(
            0,
            ber * 2.0,
            signal.shape
        )

    def render_constellation(self, noisy):

        fig, ax = plt.subplots(figsize=(5,4))

        fig.patch.set_facecolor('#050816')

        ax.set_facecolor('#111827')

        if self.scheme == "BPSK":

            ax.scatter(
                noisy,
                np.zeros(len(noisy)),
                s=10
            )

        else:

            ax.scatter(
                noisy[:,0],
                noisy[:,1],
                s=10
            )

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

        pad = (
            self.cols - (len(bits)%self.cols)
        ) % self.cols

        bits += '0' * pad

        rows = len(bits)//self.cols

        encoded = ""

        html = "<div>"

        col_parity = ['0'] * self.cols

        for r in range(rows):

            row = bits[
                r*self.cols:(r+1)*self.cols
            ]

            p = self.parity(row)

            for i,b in enumerate(row):

                html += f'''
                <span class="matrix-cell data-bit">
                {b}
                </span>
                '''

                if b == '1':

                    col_parity[i] = (
                        '0'
                        if col_parity[i]=='1'
                        else '1'
                    )

            html += f'''
            <span class="matrix-cell parity-bit">
            {p}
            </span><br>
            '''

            encoded += row + p

        master = self.parity(
            ''.join(col_parity)
        )

        for cp in col_parity:

            html += f'''
            <span class="matrix-cell parity-bit">
            {cp}
            </span>
            '''

        html += f'''
        <span class="matrix-cell parity-bit">
        {master}
        </span>
        '''

        html += "</div>"

        encoded += ''.join(col_parity) + master

        return encoded, html, pad

# =============================================================================
# APP
# =============================================================================

st.title("📡 Multimedia Telecom Lab")
st.caption("Fuente + Canal + FEC + Modulación + AWGN")

main_tabs = st.tabs([
    "📄 Texto",
    "🖼 Imagen",
    "🎵 Audio",
    "🎬 Video"
])

# =============================================================================
# TEXTO
# =============================================================================

with main_tabs[0]:

    sub = st.tabs([
        "Fuente",
        "Canal",
        "Pipeline Completo"
    ])

    # =========================================================================
    # FUENTE
    # =========================================================================

    with sub[0]:

        st.header("📄 Codificación de Fuente")

        text = st.text_area(
            "Texto",
            "HOLA TELECOMUNICACIONES"
        )

        algo = st.selectbox(
            "Algoritmo",
            ["Huffman","RLE"]
        )

        if st.button("Codificar Fuente"):

            stats = AnalizadorTexto(text).calcular()

            c1,c2,c3 = st.columns(3)

            with c1:
                st.metric(
                    "Entropía",
                    f"{stats.entropia:.4f}"
                )

            with c2:
                st.metric(
                    "Símbolos",
                    stats.simbolos_unicos
                )

            with c3:
                st.metric(
                    "Total",
                    stats.total_simbolos
                )

            if algo == "Huffman":

                encoded, inverse, codes = (
                    HuffmanCoder().encode(text)
                )

                st.success("Compresión Huffman OK")

                st.code(encoded[:300])

                st.json(codes)

                decoded = (
                    HuffmanCoder().decode(
                        encoded,
                        inverse
                    )
                )

                st.text_area(
                    "Reconstruido",
                    decoded
                )

            else:

                encoded = (
                    RLECoder().encode(text)
                )

                st.success("Compresión RLE OK")

                st.code(encoded)

                decoded = (
                    RLECoder().decode(encoded)
                )

                st.text_area(
                    "Reconstruido",
                    decoded
                )

    # =========================================================================
    # CANAL
    # =========================================================================

    with sub[1]:

        st.header("📡 Codificación de Canal")

        bits = st.text_area(
            "Bits",
            "101010101010111100001111"
        )

        mod = st.selectbox(
            "Modulación",
            ["BPSK","QPSK","QAM16"]
        )

        fec_cols = st.selectbox(
            "FEC",
            [4,8,16]
        )

        ber = st.slider(
            "BER",
            0.0,
            1.0,
            0.05
        )

        if st.button("Transmitir Canal"):

            fec = MatrixFEC(fec_cols)

            fec_bits, html, pad = (
                fec.encode(bits)
            )

            st.subheader("FEC 2D")

            st.markdown(
                html,
                unsafe_allow_html=True
            )

            st.code(fec_bits[:500])

            modulator = Modulator(mod)

            signal = modulator.modulate(
                fec_bits
            )

            noisy = modulator.channel_awgn(
                signal,
                ber
            )

            st.subheader("Constelación")

            st.pyplot(
                modulator.render_constellation(
                    noisy
                )
            )

    # =========================================================================
    # PIPELINE
    # =========================================================================

    with sub[2]:

        st.header("🔥 Pipeline Completo")

        text = st.text_area(
            "Payload",
            "HOLA MUNDO"
        )

        source_algo = st.selectbox(
            "Fuente",
            ["Huffman","RLE"]
        )

        mod = st.selectbox(
            "Modulación",
            ["BPSK","QPSK","QAM16"],
            key="pipe_mod"
        )

        fec_cols = st.selectbox(
            "FEC",
            [4,8,16],
            key="pipe_fec"
        )

        ber = st.slider(
            "BER AWGN",
            0.0,
            1.0,
            0.02,
            key="pipe_ber"
        )

        if st.button("Ejecutar Pipeline Completo"):

            # =============================================================
            # FUENTE
            # =============================================================

            if source_algo == "Huffman":

                encoded, inverse, codes = (
                    HuffmanCoder().encode(text)
                )

                source_bits = encoded

            else:

                rle = RLECoder()

                rle_encoded = rle.encode(text)

                source_bits = bytes_to_bits(
                    rle_encoded.encode()
                )

            st.success("Fuente codificada")

            st.code(source_bits[:500])

            # =============================================================
            # FEC
            # =============================================================

            fec = MatrixFEC(fec_cols)

            fec_bits, html, pad = (
                fec.encode(source_bits)
            )

            st.subheader("FEC Matricial")

            st.markdown(
                html,
                unsafe_allow_html=True
            )

            # =============================================================
            # MODULACION
            # =============================================================

            modulator = Modulator(mod)

            signal = modulator.modulate(
                fec_bits
            )

            noisy = modulator.channel_awgn(
                signal,
                ber
            )

            st.subheader("Canal AWGN")

            st.pyplot(
                modulator.render_constellation(
                    noisy
                )
            )

            # =============================================================
            # PAYLOAD
            # =============================================================

            payload = {

                "algoritmo_fuente": source_algo,
                "modulacion": mod,
                "fec_cols": fec_cols,
                "padding": pad,

                "rx_bits": inject_bit_errors(
                    fec_bits,
                    ber
                )
            }

            if source_algo == "Huffman":

                payload["inverse"] = inverse

            st.markdown(
                create_download_link(
                    payload,
                    "pipeline.bin"
                ),
                unsafe_allow_html=True
            )

            st.success("Pipeline completo ejecutado")

# =============================================================================
# IMAGEN
# =============================================================================

with main_tabs[1]:

    st.header("🖼 Procesamiento Imagen")

    img = st.file_uploader(
        "Sube Imagen",
        type=["png","jpg","jpeg"]
    )

    if img:

        image = Image.open(img)

        st.image(image)

# =============================================================================
# AUDIO
# =============================================================================

with main_tabs[2]:

    st.header("🎵 Audio")

    aud = st.file_uploader(
        "Sube WAV",
        type=["wav"]
    )

    if aud:

        st.audio(aud)

# =============================================================================
# VIDEO
# =============================================================================

with main_tabs[3]:

    st.header("🎬 Video")

    vid = st.file_uploader(
        "Sube Video",
        type=["mp4"]
    )

    if vid:

        st.video(vid)
