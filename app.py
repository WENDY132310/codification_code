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

h1,h2,h3,h4{
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

.matrix-container {
    background: #0f172a;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #334155;
    margin-bottom: 20px;
    display: inline-block;
}

.matrix-cell {
    display:inline-block;
    width:32px;
    height:32px;
    line-height:32px;
    text-align:center;
    border:1px solid #334155;
    margin:2px;
    font-family:monospace;
    font-size: 16px;
    border-radius: 4px;
}

.data-bit {
    background:#1d4ed8;
    color: white;
}

.parity-bit {
    background:#059669; /* Verde para destacar paridad */
    color:white;
    font-weight:bold;
    box-shadow: 0 0 8px #059669;
}

.master-bit {
    background:#d97706; /* Naranja para maestro */
    color:white;
    font-weight:bold;
    box-shadow: 0 0 8px #d97706;
}

.empty-bit {
    background: transparent;
    border: 1px dashed #475569;
    color: transparent;
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
        bit_list[idx] = '1' if bit_list[idx] == '0' else '0'
    return "".join(bit_list)

def create_download_link(payload_dict, filename):
    json_str = json.dumps(payload_dict)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'''
    <a href="data:application/octet-stream;base64,{b64}"
    download="{filename}" style="display:inline-block; padding:10px 20px; background:#7c3aed; color:white; text-decoration:none; border-radius:8px; font-weight:bold; margin-top:10px;">
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
        probs = {k:v/total for k,v in freq.items()}
        H = -sum(p * math.log2(p) for p in probs.values() if p > 0)
        return EstadisticasInfo(
            entropia=H,
            simbolos_unicos=len(freq),
            total_simbolos=total
        )

# =============================================================================
# HUFFMAN & RLE
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
        heap = [HuffNode(v, k) for k,v in freq.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            l = heapq.heappop(heap)
            r = heapq.heappop(heap)
            heapq.heappush(heap, HuffNode(l.freq + r.freq, None, l, r))

        codes = {}
        def gen(node, current=''):
            if node.symbol is not None:
                codes[node.symbol] = current if current else '0'
            if node.left: gen(node.left, current+'0')
            if node.right: gen(node.right, current+'1')

        if heap: gen(heap[0])
        encoded = ''.join(codes[ch] for ch in text)
        inverse = {v:k for k,v in codes.items()}
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

class RLECoder:
    def encode(self, text):
        if not text: return ""
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
            return np.array([1 if b == '1' else -1 for b in bits])
        elif self.scheme == "QPSK":
            return np.array([
                {'00':(-1,-1), '01':(-1,1), '10':(1,-1), '11':(1,1)}[bits[i:i+2].ljust(2,'0')]
                for i in range(0,len(bits),2)
            ])
        else:
            levels = [-3,-1,1,3]
            return np.array([
                (levels[int(bits[i:i+4].ljust(4,'0')[:2],2)], levels[int(bits[i:i+4].ljust(4,'0')[2:],2)])
                for i in range(0,len(bits),4)
            ])

    def channel_awgn(self, signal, ber):
        return signal + np.random.normal(0, ber * 2.0, signal.shape)

    def render_constellation(self, noisy):
        fig, ax = plt.subplots(figsize=(5,4))
        fig.patch.set_facecolor('#050816')
        ax.set_facecolor('#111827')
        
        ax.axhline(0, color='#334155', linewidth=1)
        ax.axvline(0, color='#334155', linewidth=1)

        if self.scheme == "BPSK":
            ax.scatter(noisy, np.zeros(len(noisy)), s=15, c='#00ffff', alpha=0.7)
        else:
            ax.scatter(noisy[:,0], noisy[:,1], s=15, c='#00ffff', alpha=0.7)
        return fig

# =============================================================================
# FEC 2D CON EXPLICACION PASO A PASO
# =============================================================================

class MatrixFEC:
    def __init__(self, cols=4):
        self.cols = cols

    def parity(self, bits):
        # Paridad par: retorna '1' si hay un número impar de unos, '0' si es par.
        return '1' if bits.count('1') % 2 != 0 else '0'

    def encode_and_explain(self, bits):
        """Genera los bits codificados y renderiza visualmente el proceso FEC."""
        
        # 1. Padding
        pad = (self.cols - (len(bits)%self.cols)) % self.cols
        bits_padded = bits + '0' * pad
        rows = len(bits_padded) // self.cols

        encoded_bits = ""
        col_parity = ['0'] * self.cols
        
        # Estructuras de datos para la visualización
        matrix_rows = []
        row_parities = []

        # Cálculo
        for r in range(rows):
            row = bits_padded[r*self.cols : (r+1)*self.cols]
            p = self.parity(row)
            
            matrix_rows.append(row)
            row_parities.append(p)
            
            # Acumular paridad de columna
            for i, b in enumerate(row):
                if b == '1':
                    col_parity[i] = '0' if col_parity[i]=='1' else '1'
                    
            encoded_bits += row + p

        master = self.parity(''.join(col_parity))
        encoded_bits += ''.join(col_parity) + master

        # ==========================================
        # RENDERIZADO VISUAL (PASO A PASO)
        # ==========================================
        
        st.markdown("### 🧩 Construcción de la Matriz FEC (Explicación Visual)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Paso 1: Matriz de Datos Crudos")
            st.caption("Los bits se acomodan en filas. Si faltan bits al final, se rellenan con '0' (Padding).")
            html1 = '<div class="matrix-container">'
            for row in matrix_rows:
                for bit in row:
                    html1 += f'<span class="matrix-cell data-bit">{bit}</span>'
                html1 += f'<span class="matrix-cell empty-bit">x</span><br>'
            html1 += "</div>"
            st.markdown(html1, unsafe_allow_html=True)

        with col2:
            st.markdown("#### Paso 2: Paridad de Fila (Row Parity)")
            st.caption("Se cuenta la cantidad de '1's en cada fila. Se añade un bit para que el total de '1's sea PAR.")
            html2 = '<div class="matrix-container">'
            for i, row in enumerate(matrix_rows):
                for bit in row:
                    html2 += f'<span class="matrix-cell data-bit">{bit}</span>'
                html2 += f'<span class="matrix-cell parity-bit">{row_parities[i]}</span><br>'
            html2 += "</div>"
            st.markdown(html2, unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### Paso 3: Paridad de Columna")
            st.caption("Se realiza el mismo conteo de paridad, pero leyendo los bits de arriba hacia abajo (columnas).")
            html3 = '<div class="matrix-container">'
            for i, row in enumerate(matrix_rows):
                for bit in row:
                    html3 += f'<span class="matrix-cell data-bit">{bit}</span>'
                html3 += f'<span class="matrix-cell parity-bit">{row_parities[i]}</span><br>'
            
            for cp in col_parity:
                html3 += f'<span class="matrix-cell parity-bit">{cp}</span>'
            html3 += f'<span class="matrix-cell empty-bit">x</span>'
            html3 += "</div>"
            st.markdown(html3, unsafe_allow_html=True)

        with col4:
            st.markdown("#### Paso 4: Paridad Maestra (Final)")
            st.caption("El bit de la esquina inferior derecha valida tanto la fila de paridad como la columna de paridad. Si hay un solo error en la transmisión, las coordenadas (Fila, Columna) permitirán encontrarlo y corregirlo.")
            html4 = '<div class="matrix-container">'
            for i, row in enumerate(matrix_rows):
                for bit in row:
                    html4 += f'<span class="matrix-cell data-bit">{bit}</span>'
                html4 += f'<span class="matrix-cell parity-bit">{row_parities[i]}</span><br>'
            
            for cp in col_parity:
                html4 += f'<span class="matrix-cell parity-bit">{cp}</span>'
            html4 += f'<span class="matrix-cell master-bit">{master}</span>'
            html4 += "</div>"
            st.markdown(html4, unsafe_allow_html=True)

        return encoded_bits, pad

# =============================================================================
# APP
# =============================================================================

st.title("📡 Multimedia Telecom Lab")
st.caption("Fuente + Canal + FEC + Modulación + AWGN")

main_tabs = st.tabs(["📄 Texto", "🖼 Imagen", "🎵 Audio", "🎬 Video"])

# =============================================================================
# TEXTO
# =============================================================================

with main_tabs[0]:

    sub = st.tabs(["Fuente", "Canal (FEC + AWGN)", "Pipeline Completo"])

    # =========================================================================
    # FUENTE
    # =========================================================================
    with sub[0]:
        st.header("📄 Codificación de Fuente")
        text = st.text_area("Texto", "HOLA TELECOMUNICACIONES")
        algo = st.selectbox("Algoritmo", ["Huffman","RLE"])

        if st.button("Codificar Fuente"):
            stats = AnalizadorTexto(text).calcular()
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Entropía", f"{stats.entropia:.4f}")
            with c2: st.metric("Símbolos", stats.simbolos_unicos)
            with c3: st.metric("Total", stats.total_simbolos)

            if algo == "Huffman":
                encoded, inverse, codes = HuffmanCoder().encode(text)
                st.success("Compresión Huffman OK")
                st.code(encoded[:300])
                st.json(codes)
                decoded = HuffmanCoder().decode(encoded, inverse)
                st.text_area("Reconstruido", decoded)
            else:
                encoded = RLECoder().encode(text)
                st.success("Compresión RLE OK")
                st.code(encoded)
                decoded = RLECoder().decode(encoded)
                st.text_area("Reconstruido", decoded)

    # =========================================================================
    # CANAL
    # =========================================================================
    with sub[1]:
        st.header("📡 Codificación de Canal y Análisis FEC")
        
        bits = st.text_input("Bits de Prueba", "10101010101011110000")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1: mod = st.selectbox("Modulación", ["BPSK","QPSK","QAM16"])
        with col_c2: fec_cols = st.selectbox("Columnas FEC", [4,8,16])
        
        ber = st.slider("BER (Tasa de Error)", 0.0, 1.0, 0.05)

        if st.button("Transmitir Canal"):
            fec = MatrixFEC(fec_cols)
            
            # Ejecuta la explicación visual y obtiene bits codificados
            fec_bits, pad = fec.encode_and_explain(bits)

            st.markdown(f"**Bits resultantes listos para enviar ({len(fec_bits)} bits):**")
            st.code(fec_bits)

            modulator = Modulator(mod)
            signal = modulator.modulate(fec_bits)
            noisy = modulator.channel_awgn(signal, ber)

            st.subheader("Simulación AWGN Constelación")
            st.pyplot(modulator.render_constellation(noisy))

    # =========================================================================
    # PIPELINE COMPLETO
    # =========================================================================
    with sub[2]:
        st.header("🔥 Pipeline Completo de Transmisión")

        text = st.text_area("Carga Útil (Payload)", "HOLA MUNDO DESDE BOGOTA")
        
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            source_algo = st.selectbox("Cod. Fuente", ["Huffman","RLE"])
            mod = st.selectbox("Modulación", ["BPSK","QPSK","QAM16"], key="pipe_mod")
        with c_p2:
            fec_cols = st.selectbox("Matriz FEC (Columnas)", [4,8,16], key="pipe_fec")
            ber = st.slider("Canal AWGN BER", 0.0, 1.0, 0.02, key="pipe_ber")

        if st.button("Ejecutar Pipeline y Descargar"):

            # 1. FUENTE
            if source_algo == "Huffman":
                encoded, inverse, codes = HuffmanCoder().encode(text)
                source_bits = encoded
            else:
                rle = RLECoder()
                rle_encoded = rle.encode(text)
                source_bits = bytes_to_bits(rle_encoded.encode())

            st.success("✅ 1. Fuente codificada exitosamente.")

            # 2. FEC
            fec = MatrixFEC(fec_cols)
            fec_bits, pad = fec.encode_and_explain(source_bits[:64]) # Mostramos solo un fragmento para no saturar la UI
            st.caption("Nota: La visualización muestra solo los primeros bloques para ilustrar el proceso.")

            # Codificamos la cadena completa internamente
            full_fec_bits, _ = MatrixFEC(fec_cols).encode_and_explain(source_bits) if False else ("", pad) # hack para evitar doble render
            
            # Helper for clean silent encoding of full payload
            def silent_fec(b, c):
                pd = (c - (len(b)%c)) % c
                bp = b + '0'*pd
                rs = len(bp)//c
                enc = ""
                cp = ['0']*c
                for r in range(rs):
                    row = bp[r*c:(r+1)*c]
                    p = '1' if row.count('1')%2!=0 else '0'
                    for i,bit in enumerate(row):
                        if bit=='1': cp[i] = '0' if cp[i]=='1' else '1'
                    enc += row + p
                master = '1' if ''.join(cp).count('1')%2!=0 else '0'
                enc += ''.join(cp) + master
                return enc

            full_fec_bits = silent_fec(source_bits, fec_cols)
            st.success("✅ 2. Capa FEC inyectada en la trama principal.")

            # 3. MODULACION
            modulator = Modulator(mod)
            signal = modulator.modulate(full_fec_bits)
            noisy = modulator.channel_awgn(signal, ber)

            st.subheader("Ruido en el Canal (Constelación RX)")
            st.pyplot(modulator.render_constellation(noisy))

            # 4. EMPAQUETADO Y DESCARGA
            payload = {
                "algoritmo_fuente": source_algo,
                "modulacion": mod,
                "fec_cols": fec_cols,
                "padding": pad,
                "rx_bits": inject_bit_errors(full_fec_bits, ber)
            }
            if source_algo == "Huffman":
                payload["inverse"] = inverse

            st.success("✅ 3. Simulación completada. El archivo `.bin` contiene la trama con los errores inyectados listos para que el receptor los corrija usando la matriz FEC.")
            
            st.markdown(
                create_download_link(payload, "pipeline_rx.bin"),
                unsafe_allow_html=True
            )

# =============================================================================
# IMAGEN / AUDIO / VIDEO (Dummies)
# =============================================================================
with main_tabs[1]:
    st.header("🖼 Procesamiento Imagen")
    img = st.file_uploader("Sube Imagen", type=["png","jpg","jpeg"])
    if img: st.image(Image.open(img))

with main_tabs[2]:
    st.header("🎵 Audio")
    aud = st.file_uploader("Sube WAV", type=["wav"])
    if aud: st.audio(aud)

with main_tabs[3]:
    st.header("🎬 Video")
    vid = st.file_uploader("Sube Video", type=["mp4"])
    if vid: st.video(vid)
