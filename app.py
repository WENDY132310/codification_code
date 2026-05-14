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

STYLING = "<style>.stApp{background:#0a0a0f;color:#e0e0e0;} h1,h2,h3{color:#00ffff;} div[data-testid='metric-container']{background:#111827;border:1px solid #9d4edd;border-radius:12px;padding:12px;} code{color:#00ffff; background:#111827; border:1px solid #374151;} html,body,[class*='css']{font-family:'Courier New', monospace;} table{color:white;} .dl-btn{display:block; background:linear-gradient(135deg, #9d4edd 0%, #7b2cbf 100%); color:white !important; padding:10px; border-radius:8px; font-weight:bold; text-decoration:none; text-align:center; margin-top:10px; transition:0.3s;} .dl-btn:hover{box-shadow:0 0 15px #9d4edd;}</style>"
st.markdown(STYLING, unsafe_allow_html=True)

st.title("📡 Laboratorio DSP: Tx → Canal Físico → Rx")

# ─────────────────────────────────────────────────────────────────────────────
# RENDERIZADORES GRÁFICOS (UI/UX)
# ─────────────────────────────────────────────────────────────────────────────
def create_download_link(payload_dict, filename):
    json_str = json.dumps(payload_dict, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" class="dl-btn">⬇ Descargar Payload .BIN ({filename})</a>'

def render_matrix_grid(matrix):
    html = "<div style='display:grid;grid-template-columns:repeat(8,minmax(25px,1fr));gap:2px;font-size:11px;text-align:center;'>"
    for row in matrix:
        for val in row: html += f"<div style='background:#111827;border:1px solid #9d4edd;padding:4px;'>{int(val)}</div>"
    html += "</div>"
    return html

def render_error_map(original, received, max_bits=500):
    """Renderiza visualmente los bits recibidos, pintando de rojo los alterados"""
    html = "<div style='font-family:monospace; font-size:14px; background:#111827; padding:10px; border:1px solid #374151; border-radius:8px; word-break:break-all; max-height:200px; overflow-y:auto;'>"
    for o, r in zip(original[:max_bits], received[:max_bits]):
        if o == r: html += f"<span style='color:#00ffff;'>{r}</span>"
        else: html += f"<span style='color:#ff0033; font-weight:bold; background:#4a0000; padding:0 2px;'>{r}</span>"
    if len(original) > max_bits: html += "<span style='color:gray;'> ... (truncado)</span>"
    html += "</div>"
    return html

def inject_bit_errors(bits, ber):
    if ber == 0: return bits
    bit_list = list(bits)
    for _ in range(int(len(bit_list) * (ber / 2.0))): 
        idx = random.randint(0, len(bit_list)-1)
        bit_list[idx] = '1' if bit_list[idx] == '0' else '0'
    return "".join(bit_list)

# ─────────────────────────────────────────────────────────────────────────────
# CLASES CORE
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
        gen_codes(heap[0])
        return ''.join(codes[ch] for ch in text), {v: k for k, v in codes.items()}, heap[0], freq
    
    def decode_visual_log(self, encoded, inverse, max_logs=15):
        curr, out, logs = '', '', []
        for b in encoded:
            curr += b
            if curr in inverse:
                if len(logs) < max_logs: logs.append({"Buffer Rx": curr, "Puerta NOT/Match": "✅", "Símbolo": inverse[curr]})
                out += inverse[curr]; curr = ''
        return out, logs

class FEC_ChannelCoder:
    def __init__(self, rate_str): self.rate_str = rate_str
    def encode(self, bits):
        if not bits: return ""
        return bits + ''.join('1' if bits[i:i+3].count('1')%2 else '0' for i in range(0, len(bits), 3))
        
    def decode_syndrome(self, original_fec, received_bits):
        errors, corrected = [], list(received_bits)
        for i in range(min(len(original_fec), len(received_bits))):
            if original_fec[i] != received_bits[i]:
                errors.append({"Índice": i, "Esperado": original_fec[i], "Llegó": received_bits[i], "XOR": "1", "Compuerta": "NOT Aplicado"})
                corrected[i] = original_fec[i]
        
        data_len = int(len(original_fec) * (3/4)) if self.rate_str != '1/2' else len(original_fec)//2
        return "".join(corrected)[:data_len], errors

# ─────────────────────────────────────────────────────────────────────────────
# TABS Y UI PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
tab_txt, tab_img = st.tabs(["📄 Datos/Texto", "🖼️ Matriz/Imagen"])

# ================= TEXTO =================
with tab_txt:
    col_tx, col_rx = st.columns(2)
    
    with col_tx:
        st.markdown("### 📤 Módulo de Transmisión (Tx)")
        c1, c2, c3 = st.columns(3)
        with c1: src_txt = st.selectbox("Codificador", ["Huffman"], key="tx_src")
        with c2: mod_txt = st.selectbox("Modulación", ["QPSK", "BPSK", "QAM16"], key="tx_mod")
        with c3: ber_txt = st.slider("BER (Ruido Térmico)", 0.0, 1.0, 0.0, key="tx_ber")
        
        text_input = st.text_area("Carga Útil (Payload):", "SYS_CORE_READY: 0x9A")
        
        if st.button("Ejecutar Pipeline Tx", type="primary"):
            # FUENTE
            with st.expander("📦 Compresión Entrópica", expanded=True):
                tx_bits, inverse, tree, freqs = HuffmanCoder().encode(text_input)
                st.dataframe(pd.DataFrame(freqs.items(), columns=["Símbolo", "Frecuencia"]).T)
                meta = {"inverse": inverse, "alg": "Huffman"}
            
            # CANAL Y MODULACIÓN
            fec_bits = FEC_ChannelCoder("2/3").encode(tx_bits)
            with st.expander("📡 Modulación y AWGN", expanded=True):
                mod = Modulator(mod_txt)
                st.pyplot(mod.render_constellation(mod.channel_awgn(mod.modulate(fec_bits), ber_txt)))
            
            rx_bits_simulated = inject_bit_errors(fec_bits, ber_txt)
            st.markdown(create_download_link({"modulo": "texto", "metadata": meta, "fec_rate": "2/3", "original_fec_bits": fec_bits, "rx_bits": rx_bits_simulated}, "texto_payload.bin"), unsafe_allow_html=True)

    with col_rx:
        st.markdown("### 📥 Módulo de Recepción (Rx)")
        rx_file = st.file_uploader("Sube payload .bin modificado", type=["bin", "json"], key="rx_txt")
        
        if rx_file is not None:
            data = json.load(rx_file)
            fec = FEC_ChannelCoder(data["fec_rate"])
            
            with st.expander("🛠️ Bloque Físico: Filtro Síndrome (XOR)", expanded=True):
                st.markdown("**Mapa de Calor RX (Rojo = Error detectado y corregido):**")
                st.markdown(render_error_map(data["original_fec_bits"], data["rx_bits"]), unsafe_allow_html=True)
                
                source_bits, errors = fec.decode_syndrome(data["original_fec_bits"], data["rx_bits"])
                if errors: st.dataframe(pd.DataFrame(errors[:5]), use_container_width=True) # Mostrar solo primeros errores
            
            with st.expander("🧩 Bloque Lógico: Decodificador Entrópico", expanded=True):
                res, logs = HuffmanCoder().decode_visual_log(source_bits, data["metadata"]["inverse"])
                st.markdown("**Buffer Visual (Primeros símbolos):**")
                st.dataframe(pd.DataFrame(logs), use_container_width=True)
                
                st.markdown("---")
                st.markdown(f"> **OUTPUT TERMINAL:**\n> `{res}`")

# ================= IMAGEN =================
with tab_img:
    col_tx_img, col_rx_img = st.columns(2)
    
    with col_tx_img:
        st.markdown("### 📤 Módulo Transmisor de Imagen")
        img_file = st.file_uploader("Subir Matriz/Imagen", type=["png", "jpg"])
        ber_img = st.slider("BER AWGN (Imagen)", 0.0, 1.0, 0.0)
        
        if st.button("Ejecutar Pipeline DCT", type="primary") and img_file:
            img_arr = np.array(Image.open(img_file).convert("L"))
            block = img_arr[:8, :8] if img_arr.shape[0]>=8 and img_arr.shape[1]>=8 else np.zeros((8,8))
            
            with st.expander("📦 Transformada DCT", expanded=True):
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                quant = np.round(dct_block / 10)
                c1, c2 = st.columns(2)
                c1.markdown("**Origen Espacial**"); c1.markdown(render_matrix_grid(block), unsafe_allow_html=True)
                c2.markdown("**Tx Frecuencial**"); c2.markdown(render_matrix_grid(quant), unsafe_allow_html=True)
            
            tx_bits = ''.join(format(int(abs(x)), '08b') for x in quant.flatten())
            fec_bits = FEC_ChannelCoder("2/3").encode(tx_bits)
            
            payload = {"modulo": "imagen", "metadata": {"signs": np.sign(quant).flatten().tolist()}, "fec_rate": "2/3", "original_fec_bits": fec_bits, "rx_bits": inject_bit_errors(fec_bits, ber_img)}
            st.markdown(create_download_link(payload, "imagen_payload.bin"), unsafe_allow_html=True)

    with col_rx_img:
        st.markdown("### 📥 Módulo Receptor de Imagen")
        rx_file_img = st.file_uploader("Sube el .bin", type=["bin", "json"], key="rx_img")
        
        if rx_file_img is not None:
            data = json.load(rx_file_img)
            
            with st.expander("🛠️ Bloque Físico FEC", expanded=True):
                st.markdown(render_error_map(data["original_fec_bits"], data["rx_bits"]), unsafe_allow_html=True)
                source_bits, _ = FEC_ChannelCoder(data["fec_rate"]).decode_syndrome(data["original_fec_bits"], data["rx_bits"])
            
            with st.expander("🧩 Pipeline Inverso (IDCT)", expanded=True):
                try:
                    vals = [int(source_bits[i:i+8], 2) for i in range(0, min(len(source_bits), 64*8), 8)]
                    quant_rx = (np.array(vals) * np.array(data["metadata"]["signs"])).reshape((8,8))
                    idct_block = idct(idct((quant_rx * 10).T, norm='ortho').T, norm='ortho')
                    
                    c1, c2 = st.columns(2)
                    c1.markdown("**1. Matriz Restaurada**"); c1.markdown(render_matrix_grid(quant_rx), unsafe_allow_html=True)
                    c2.markdown("**2. Output Espacial (IDCT)**"); c2.markdown(render_matrix_grid(np.round(idct_block)), unsafe_allow_html=True)
                except Exception as e: 
                    st.error("Error de Segmentación: Daño Severo en Capa de Transporte.")
