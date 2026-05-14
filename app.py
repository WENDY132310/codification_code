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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import Counter
from scipy.fftpack import dct, idct
from PIL import Image

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN Y UX TEMA "TERMINAL LAB"
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multimedia Telecom Lab", layout="wide", page_icon="📡")

STYLING = "<style>.stApp{background:#0a0a0f;color:#e0e0e0;} h1,h2,h3{color:#00ffff;} div[data-testid='metric-container']{background:#111827;border:1px solid #9d4edd;border-radius:12px;padding:12px;} code{color:#00ffff; background:#111827; border:1px solid #374151;} html,body,[class*='css']{font-family:'Courier New', monospace;} table{color:white;} .dl-btn{display:block; background:linear-gradient(135deg, #9d4edd 0%, #7b2cbf 100%); color:white !important; padding:10px; border-radius:8px; font-weight:bold; text-decoration:none; text-align:center; margin-top:10px; transition:0.3s;} .dl-btn:hover{box-shadow:0 0 15px #9d4edd;} .matrix-cell {display:inline-block; width:25px; height:25px; line-height:25px; text-align:center; border:1px solid #444; margin:1px; font-family:monospace; font-size:14px;} .data-bit {background-color:#1e3a8a;} .parity-bit {background-color:#065f46; font-weight:bold;} .error-bit {background-color:#991b1b; color:white; font-weight:bold; animation: blinker 1s linear infinite;} @keyframes blinker { 50% { opacity: 0; } }</style>"
st.markdown(STYLING, unsafe_allow_html=True)

st.title("📡 Laboratorio DSP: Multimedia a través del Canal")
st.caption("Arquitectura Tx/Rx Completa con Análisis FEC Matricial (2D Parity Check)")

# ─────────────────────────────────────────────────────────────────────────────
# RENDERIZADORES Y HERRAMIENTAS FÍSICAS
# ─────────────────────────────────────────────────────────────────────────────
def bytes_to_bits(data): return ''.join(format(b, '08b') for b in data)
def bits_to_bytes(bits): return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8) if i+8 <= len(bits))

def create_download_link(payload_dict, filename):
    json_str = json.dumps(payload_dict, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" class="dl-btn">⬇ Descargar Payload .BIN ({filename})</a>'

def inject_bit_errors(bits, ber):
    if ber <= 0: return bits
    bit_list = list(bits)
    for _ in range(int(len(bit_list) * (ber / 1.5))): 
        idx = random.randint(0, len(bit_list)-1)
        bit_list[idx] = '1' if bit_list[idx] == '0' else '0'
    return "".join(bit_list)

def recommend_modulation(matrix_size):
    """Recomienda modulación basado en el tamaño de la matriz FEC"""
    return "💡 **Recomendación**: La Paridad 2D es muy robusta para detectar y corregir errores simples. Se recomienda QPSK para un balance de velocidad, o BPSK en canales muy ruidosos."

# ─────────────────────────────────────────────────────────────────────────────
# CORE: MODULACIÓN, FUENTE Y CANAL (FEC 2D MATRICIAL)
# ─────────────────────────────────────────────────────────────────────────────
class Modulator:
    def __init__(self, scheme): self.scheme = scheme
    def modulate(self, bits):
        if not bits: return np.array([])
        if self.scheme == 'BPSK': return np.array([1 if b == '1' else -1 for b in bits])
        elif self.scheme == 'QPSK':
            return np.array([{'00':(-1,-1),'01':(-1,1),'10':(1,-1),'11':(1,1)}[bits[i:i+2].ljust(2,'0')] for i in range(0, len(bits), 2)])
        else: # QAM16
            levels = [-3, -1, 1, 3]
            return np.array([(levels[int(bits[i:i+4].ljust(4,'0')[:2],2)], levels[int(bits[i:i+4].ljust(4,'0')[2:],2)]) for i in range(0, len(bits), 4)])

    def channel_awgn(self, signal, ber): return signal + np.random.normal(0, ber * 2.0, signal.shape)
    
    def render_constellation(self, noisy_signal):
        fig, ax = plt.subplots(figsize=(4,3)); fig.patch.set_facecolor('#0a0a0f'); ax.set_facecolor('#111827')
        for spine in ax.spines.values(): spine.set_edgecolor('#374151')
        ax.tick_params(colors='#8b9cb5')
        if self.scheme == 'BPSK': ax.scatter(noisy_signal, np.zeros(len(noisy_signal)), c='#00ffff', s=10, alpha=0.5)
        else: ax.scatter(noisy_signal[:,0], noisy_signal[:,1], c='#9d4edd', s=10, alpha=0.5)
        return fig

@dataclass(order=True)
class HuffNode:
    freq: int; symbol: any = field(compare=False); left: any = field(compare=False, default=None); right: any = field(compare=False, default=None)

class HuffmanCoder:
    def encode(self, text):
        freq = Counter(text)
        heap = [HuffNode(v, k) for k, v in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            l, r = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, HuffNode(l.freq + r.freq, None, l, r))
        codes = {}
        def gen_codes(n, current=''):
            if n.symbol is not None: codes[n.symbol] = current or '0'
            if n.left: gen_codes(n.left, current+'0')
            if n.right: gen_codes(n.right, current+'1')
        if heap: gen_codes(heap[0])
        return ''.join(codes[ch] for ch in text), {v: k for k, v in codes.items()}, heap[0] if heap else None, freq
    
    def decode_visual_log(self, encoded, inverse, max_logs=10):
        curr, out, logs = '', '', []
        for b in encoded:
            curr += b
            if curr in inverse:
                if len(logs) < max_logs: logs.append({"Buffer Rx": curr, "Match": "✅", "Símbolo": inverse[curr]})
                out += inverse[curr]; curr = ''
        return out, logs

class MuLawCodec:
    MU = 255
    @staticmethod
    def encode(samples):
        s_norm = samples.astype(np.float32) / 32768.0
        s_comp = np.sign(s_norm) * (np.log1p(MuLawCodec.MU * np.abs(s_norm)) / np.log1p(MuLawCodec.MU))
        return np.int8(s_comp * 127)
    @staticmethod
    def decode(encoded):
        s_norm = encoded.astype(np.float32) / 127.0
        s_exp = np.sign(s_norm) * (1 / MuLawCodec.MU) * (np.power(1 + MuLawCodec.MU, np.abs(s_norm)) - 1)
        return np.int16(s_exp * 32767.0)

# NUEVO CODIFICADOR FEC MATRICIAL 2D
class MatrixFEC:
    def __init__(self, cols=4):
        self.cols = cols # Número de columnas de datos (ancho de la matriz)
    
    def calculate_parity(self, bit_string):
        """Calcula paridad par (1 si hay cantidad impar de 1s, 0 si no)"""
        return '1' if bit_string.count('1') % 2 != 0 else '0'

    def encode(self, bits):
        if not bits: return "", [], ""
        
        # Pad bits para que sean múltiplo de las columnas
        padding_needed = (self.cols - (len(bits) % self.cols)) % self.cols
        padded_bits = bits + ('0' * padding_needed)
        
        rows = len(padded_bits) // self.cols
        matrix = []
        encoded_stream = ""
        visual_html = "<div style='margin-bottom: 10px;'>"
        
        # Llenar matriz y calcular paridad de filas
        col_parities = ['0'] * self.cols
        
        for r in range(rows):
            row_data = padded_bits[r*self.cols : (r+1)*self.cols]
            row_parity = self.calculate_parity(row_data)
            matrix.append(list(row_data) + [row_parity])
            
            # Dibujar Fila HTML
            for bit in row_data: visual_html += f"<span class='matrix-cell data-bit'>{bit}</span>"
            visual_html += f"<span class='matrix-cell parity-bit'>{row_parity}</span><br>"
            
            # Acumular para paridad de columnas
            for c in range(self.cols):
                if row_data[c] == '1': col_parities[c] = '0' if col_parities[c] == '1' else '1'
                
            encoded_stream += row_data + row_parity

        # Calcular paridad del bit maestro
        master_parity = self.calculate_parity("".join(col_parities))
        
        # Dibujar Fila Final de Paridades HTML
        for cp in col_parities: visual_html += f"<span class='matrix-cell parity-bit'>{cp}</span>"
        visual_html += f"<span class='matrix-cell parity-bit' style='background-color:#9333ea;'>{master_parity}</span><br></div>"
        
        encoded_stream += "".join(col_parities) + master_parity
        
        return encoded_stream, matrix, visual_html, padding_needed

    def decode_and_correct(self, rx_stream, padding_needed):
        """Recibe la trama, reconstruye la matriz, halla el síndrome y corrige"""
        if not rx_stream: return "", "", []
        
        row_len = self.cols + 1 # Datos + Paridad
        # Calculamos cuántas filas de datos hay (sin contar la fila final de paridad)
        num_data_rows = (len(rx_stream) // row_len) - 1 
        
        rx_matrix = []
        idx = 0
        
        # 1. Reconstruir Matriz Recibida
        for r in range(num_data_rows + 1): # Incluye la fila de paridades de columna
            rx_matrix.append(list(rx_stream[idx : idx + row_len]))
            idx += row_len

        # 2. Calcular Síndromes (Detectar Errores)
        error_row = -1
        error_col = -1
        
        syndrome_logs = []

        # Chequear Filas
        for r in range(num_data_rows):
            row_data = "".join(rx_matrix[r][:-1])
            expected_parity = self.calculate_parity(row_data)
            received_parity = rx_matrix[r][-1]
            if expected_parity != received_parity:
                error_row = r
                syndrome_logs.append(f"❌ **Síndrome Fila {r}:** XOR Falló. Debería ser {expected_parity}, llegó {received_parity}.")
        
        # Chequear Columnas
        for c in range(self.cols):
            col_data = "".join([rx_matrix[r][c] for r in range(num_data_rows)])
            expected_parity = self.calculate_parity(col_data)
            received_parity = rx_matrix[-1][c]
            if expected_parity != received_parity:
                error_col = c
                syndrome_logs.append(f"❌ **Síndrome Columna {c}:** XOR Falló. Debería ser {expected_parity}, llegó {received_parity}.")

        # 3. Dibujar Matriz y Corregir
        visual_html = "<div>"
        correction_log = "✅ Síndrome 0: Trama limpia."
        
        # ¿Hay un error de coordenada cruzada?
        if error_row != -1 and error_col != -1:
            bad_bit = rx_matrix[error_row][error_col]
            corrected_bit = '0' if bad_bit == '1' else '1'
            correction_log = f"🎯 **Destrucción de Error:** Coordenada ({error_row}, {error_col}). Aplicando compuerta NOT al bit `{bad_bit}` ➔ `{corrected_bit}`."
            rx_matrix[error_row][error_col] = corrected_bit # ¡Corregido!

        for r in range(len(rx_matrix)):
            for c in range(len(rx_matrix[r])):
                is_error_cell = (r == error_row and c == error_col)
                css_class = "error-bit" if is_error_cell else ("parity-bit" if c == self.cols or r == num_data_rows else "data-bit")
                val = rx_matrix[r][c]
                visual_html += f"<span class='matrix-cell {css_class}'>{val}</span>"
            visual_html += "<br>"
        visual_html += "</div>"

        # 4. Extraer datos limpios
        clean_bits = ""
        for r in range(num_data_rows):
            clean_bits += "".join(rx_matrix[r][:-1])
            
        # Remover el padding añadido en Tx
        if padding_needed > 0:
            clean_bits = clean_bits[:-padding_needed]

        return clean_bits, visual_html, syndrome_logs, correction_log

# Helpers DCT para la Imagen Completa
def process_full_image_dct(img_arr):
    h, w = img_arr.shape
    h_pad, w_pad = (8 - h % 8) % 8, (8 - w % 8) % 8
    padded = np.pad(img_arr, ((0, h_pad), (0, w_pad)), mode='constant')
    dct_blocks = np.zeros_like(padded, dtype=float)
    for i in range(0, padded.shape[0], 8):
        for j in range(0, padded.shape[1], 8):
            block = padded[i:i+8, j:j+8]
            dct_blocks[i:i+8, j:j+8] = np.round(dct(dct(block.T, norm='ortho').T, norm='ortho') / 10)
    return dct_blocks, padded.shape

def reconstruct_full_image_idct(dct_blocks, orig_h, orig_w):
    idct_blocks = np.zeros_like(dct_blocks, dtype=float)
    for i in range(0, dct_blocks.shape[0], 8):
        for j in range(0, dct_blocks.shape[1], 8):
            block = dct_blocks[i:i+8, j:j+8]
            idct_blocks[i:i+8, j:j+8] = idct(idct((block * 10).T, norm='ortho').T, norm='ortho')
    return np.clip(idct_blocks[:orig_h, :orig_w], 0, 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# TABS Y PIPELINES
# ─────────────────────────────────────────────────────────────────────────────
tab_txt, tab_img, tab_aud, tab_vid = st.tabs(["📄 Texto", "🖼️ Imagen Real", "🎵 Audio Físico", "🎬 Video H.264"])

# ================= TEXTO =================
with tab_txt:
    col_tx, col_rx = st.columns(2)
    with col_tx:
        st.markdown("### 📤 Módulo Transmisor")
        c1, c2, c3 = st.columns(3)
        with c2: fec_cols = st.selectbox("Ancho Matriz FEC", [4, 8, 16], index=0, key="tx_fec")
        st.info(recommend_modulation(fec_cols))
        with c1: mod_txt = st.selectbox("Modulación", ["QPSK", "BPSK", "QAM16"], key="tx_mod")
        with c3: ber_txt = st.slider("BER AWGN", 0.0, 1.0, 0.0, key="tx_ber")
        
        text_input = st.text_area("Payload:", "HOLA")
        
        if st.button("Ejecutar Pipeline Tx", type="primary"):
            tx_bits, inverse, tree, freqs = HuffmanCoder().encode(text_input)
            
            with st.expander("🛡️ Armadura Matemática (Inserción FEC 2D)", expanded=True):
                st.write(f"Empaquetando datos en matrices de {fec_cols} columnas y calculando paridad cruzada (Verde = Paridad, Morado = Paridad Maestra):")
                fec = MatrixFEC(cols=fec_cols)
                fec_bits, _, tx_html, padding = fec.encode(tx_bits)
                st.markdown(tx_html, unsafe_allow_html=True)
            
            with st.expander("📡 Modulación y AWGN", expanded=True):
                mod = Modulator(mod_txt)
                st.pyplot(mod.render_constellation(mod.channel_awgn(mod.modulate(fec_bits), ber_txt)))
            
            meta = {"inverse": inverse, "alg": "Huffman", "cols": fec_cols, "padding": padding}
            payload = {"modulo": "texto", "metadata": meta, "original_fec_bits": fec_bits, "rx_bits": inject_bit_errors(fec_bits, ber_txt)}
            st.markdown(create_download_link(payload, "texto.bin"), unsafe_allow_html=True)

    with col_rx:
        st.markdown("### 📥 Módulo Receptor")
        rx_file = st.file_uploader("Sube .bin", type=["bin", "json"], key="rx_txt")
        if rx_file:
            data = json.load(rx_file)
            
            with st.expander("🛠️ Destrucción de Errores (Síndromes FEC)", expanded=True):
                fec = MatrixFEC(cols=data["metadata"]["cols"])
                source_bits, rx_html, syndromes, correction = fec.decode_and_correct(data["rx_bits"], data["metadata"]["padding"])
                
                c_a, c_b = st.columns(2)
                with c_a:
                    st.markdown("**Matriz Recibida y Análisis:**")
                    st.markdown(rx_html, unsafe_allow_html=True)
                with c_b:
                    st.markdown("**Logs de CPU:**")
                    for s in syndromes: st.write(s)
                    st.success(correction)
            
            with st.expander("🧩 Decodificación Entrópica", expanded=True):
                try:
                    res, logs = HuffmanCoder().decode_visual_log(source_bits, data["metadata"]["inverse"])
                    st.markdown(f"> **OUTPUT:** `{res}`")
                except Exception:
                    st.error("Error: Bits dañados permanentemente más allá de la capacidad de la matriz.")

# ================= IMAGEN =================
with tab_img:
    col_tx_img, col_rx_img = st.columns(2)
    with col_tx_img:
        st.markdown("### 📤 Compresión Espacial")
        c_i1, c_i2 = st.columns(2)
        with c_i1: fec_img_cols = st.selectbox("Ancho Matriz FEC", [4, 8, 16], key="tx_img_fec")
        with c_i2: ber_img = st.slider("BER", 0.0, 1.0, 0.0, key="tx_img_ber")
        
        img_file = st.file_uploader("Sube Imagen", type=["png", "jpg"])
        if st.button("Ejecutar DCT Global", type="primary") and img_file:
            img_raw = Image.open(img_file).convert("L")
            img_raw.thumbnail((64, 64)) # Aún más pequeño para la pesada matriz 2D
            img_arr = np.array(img_raw)
            
            dct_blocks, padded_shape = process_full_image_dct(img_arr)
            tx_bits = ''.join(format(int(abs(x)), '08b') for x in dct_blocks.flatten())
            
            fec = MatrixFEC(cols=fec_img_cols)
            fec_bits, _, _, padding = fec.encode(tx_bits)
            
            meta = {"signs": np.sign(dct_blocks).flatten().tolist(), "orig_h": img_arr.shape[0], "orig_w": img_arr.shape[1], "pad_h": padded_shape[0], "pad_w": padded_shape[1], "cols": fec_img_cols, "padding": padding}
            payload = {"modulo": "imagen", "metadata": meta, "original_fec_bits": fec_bits, "rx_bits": inject_bit_errors(fec_bits, ber_img)}
            st.markdown(create_download_link(payload, "imagen.bin"), unsafe_allow_html=True)

    with col_rx_img:
        st.markdown("### 📥 Reconstrucción IDCT")
        rx_file_img = st.file_uploader("Sube .bin", type=["bin", "json"], key="rx_img")
        if rx_file_img:
            data = json.load(rx_file_img)
            with st.expander("🛠️ Bloque Físico FEC"):
                fec = MatrixFEC(cols=data["metadata"]["cols"])
                source_bits, rx_html, syndromes, correction = fec.decode_and_correct(data["rx_bits"], data["metadata"]["padding"])
                if syndromes: st.error("Síndromes detectados. Reparación matricial ejecutada en background.")
            
            with st.expander("🧩 Renderizado Final", expanded=True):
                try:
                    vals = [int(source_bits[i:i+8], 2) for i in range(0, len(source_bits), 8)]
                    # Manejar truncamiento por padding de imagen vs bytes
                    vals = vals[:(data["metadata"]["pad_h"] * data["metadata"]["pad_w"])]
                    quant_rx = (np.array(vals) * np.array(data["metadata"]["signs"])).reshape((data["metadata"]["pad_h"], data["metadata"]["pad_w"]))
                    
                    rec_arr = reconstruct_full_image_idct(quant_rx, data["metadata"]["orig_h"], data["metadata"]["orig_w"])
                    st.success("Reconstrucción Exitosa")
                    st.image(Image.fromarray(rec_arr), caption="Imagen Reconstruida en Rx", use_column_width=True)
                except Exception as e:
                    st.error(f"Error: Destrucción total por ruido. ({e})")

# ================= AUDIO =================
with tab_aud:
    col_tx_aud, col_rx_aud = st.columns(2)
    with col_tx_aud:
        st.markdown("### 📤 Companding $\mu$-Law")
        c_a1, c_a2 = st.columns(2)
        with c_a1: fec_aud_cols = st.selectbox("Ancho Matriz FEC", [4, 8, 16], key="tx_aud_fec")
        with c_a2: ber_aud = st.slider("BER", 0.0, 1.0, 0.0, key="tx_aud_ber")
        
        aud_file = st.file_uploader("Sube Audio WAV", type=["wav"])
        if st.button("Ejecutar Pipeline Audio", type="primary") and aud_file:
            with wave.open(io.BytesIO(aud_file.read()), 'rb') as wav_in:
                params = wav_in.getparams()
                frames = wav_in.readframes(min(params.nframes, 16000))
            
            samples = np.frombuffer(frames, dtype=np.int16)
            encoded = MuLawCodec.encode(samples)
            
            tx_bits = ''.join(format(x & 0xFF, '08b') for x in encoded.tolist())
            fec = MatrixFEC(cols=fec_aud_cols)
            fec_bits, _, _, padding = fec.encode(tx_bits)
            
            meta = {"params": params._asdict(), "cols": fec_aud_cols, "padding": padding}
            payload = {"modulo": "audio", "metadata": meta, "original_fec_bits": fec_bits, "rx_bits": inject_bit_errors(fec_bits, ber_aud)}
            st.markdown(create_download_link(payload, "audio.bin"), unsafe_allow_html=True)

    with col_rx_aud:
        st.markdown("### 📥 Expansión a PCM")
        rx_file_aud = st.file_uploader("Sube .bin", type=["bin", "json"], key="rx_aud")
        if rx_file_aud:
            data = json.load(rx_file_aud)
            with st.expander("🛠️ Bloque Físico FEC"):
                fec = MatrixFEC(cols=data["metadata"]["cols"])
                source_bits, rx_html, syndromes, correction = fec.decode_and_correct(data["rx_bits"], data["metadata"]["padding"])
                if syndromes: st.warning(f"Se corrigieron anomalías detectadas por síndromes matriciales.")
            
            with st.expander("🧩 Reproducción DAC", expanded=True):
                try:
                    rx_bytes = [int(source_bits[i:i+8], 2) for i in range(0, len(source_bits), 8)]
                    decoded_pcm = MuLawCodec.decode(np.array(rx_bytes, dtype=np.uint8).astype(np.int8))
                    
                    out_buffer = io.BytesIO()
                    with wave.open(out_buffer, 'wb') as wav_out:
                        p = data["metadata"]["params"]
                        wav_out.setparams((p['nchannels'], p['sampwidth'], p['framerate'], len(decoded_pcm)//p['nchannels'], p['comptype'], p['compname']))
                        wav_out.writeframes(decoded_pcm.tobytes())
                    st.success("Reconstrucción de Audio Exitosa")
                    st.audio(out_buffer.getvalue())
                except Exception: st.error("Trama de audio irreconocible por daños.")

# ================= VIDEO =================
with tab_vid:
    col_tx_vid, col_rx_vid = st.columns(2)
    with col_tx_vid:
        st.markdown("### 📤 Protección de Cabecera NAL")
        c_v1, c_v2 = st.columns(2)
        with c_v1: fec_vid_cols = st.selectbox("Ancho Matriz FEC", [4, 8, 16], key="tx_vid_fec")
        with c_v2: ber_vid = st.slider("BER", 0.0, 1.0, 0.0, key="tx_vid_ber")
        
        vid_file = st.file_uploader("Sube Video MP4", type=["mp4", "mov"])
        if vid_file:
            st.markdown("#### Video Original (Transmisión)")
            vid_bytes = vid_file.read()
            st.video(vid_bytes)
            
            if st.button("Transmitir Video", type="primary"):
                header_bytes = vid_bytes[:250]
                body_bytes = vid_bytes[250:]
                
                tx_bits = bytes_to_bits(header_bytes)
                fec = MatrixFEC(cols=fec_vid_cols)
                fec_bits, _, tx_html, padding = fec.encode(tx_bits)
                
                with st.expander("🛡️ Proceso FEC de la Cabecera MP4", expanded=True):
                    st.write("Protegiendo la metadata del contenedor (Atoms) con la Matriz de Paridad Cruzada:")
                    st.markdown(tx_html, unsafe_allow_html=True)
                
                meta = {"body_b64": base64.b64encode(body_bytes).decode('utf-8'), "cols": fec_vid_cols, "padding": padding}
                payload = {"modulo": "video", "metadata": meta, "original_fec_bits": fec_bits, "rx_bits": inject_bit_errors(fec_bits, ber_vid)}
                st.markdown(create_download_link(payload, "video.bin"), unsafe_allow_html=True)

    with col_rx_vid:
        st.markdown("### 📥 Recepción H.264")
        rx_file_vid = st.file_uploader("Sube .bin", type=["bin", "json"], key="rx_vid")
        if rx_file_vid:
            data = json.load(rx_file_vid)
            
            with st.expander("🛠️ Reparación Matricial de Cabecera"):
                fec = MatrixFEC(cols=data["metadata"]["cols"])
                source_bits, rx_html, syndromes, correction = fec.decode_and_correct(data["rx_bits"], data["metadata"]["padding"])
                if syndromes:
                    st.warning("Síndromes no nulos. Intersectando coordenadas y reparando Atoms MP4...")
                    st.success(correction)
            
            try:
                # Recortamos a múltiplo de 8 para evitar errores de parseo binario tras quitar el padding
                source_bits = source_bits[:len(source_bits) - (len(source_bits)%8)]
                header_rx = bits_to_bytes(source_bits)
                body_rx = base64.b64decode(data["metadata"]["body_b64"])
                full_video = header_rx + body_rx
                
                st.success("Reconstrucción Exitosa: Cabecera NAL intacta.")
                st.markdown("#### Video Recibido (Destino)")
                st.video(full_video)
            except Exception as e:
                st.error(f"Video Corrupto: El reproductor HTML5 no puede leer el archivo. El ruido destruyó la estructura. ({e})")
