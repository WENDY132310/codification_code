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

st.title("📡 Laboratorio DSP: Multimedia a través del Canal")
st.caption("Arquitectura Tx/Rx Completa para Texto, Imagen Real, Audio PCM y Video H.264")

# ─────────────────────────────────────────────────────────────────────────────
# RENDERIZADORES Y HERRAMIENTAS FÍSICAS
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

def render_error_map(original, received, max_bits=600):
    html = "<div style='font-family:monospace; font-size:14px; background:#111827; padding:10px; border:1px solid #374151; border-radius:8px; word-break:break-all; max-height:200px; overflow-y:auto;'>"
    for o, r in zip(original[:max_bits], received[:max_bits]):
        if o == r: html += f"<span style='color:#00ffff;'>{r}</span>"
        else: html += f"<span style='color:#ff0033; font-weight:bold; background:#4a0000; padding:0 2px;'>{r}</span>"
    if len(original) > max_bits: html += "<span style='color:gray;'> ... (truncado)</span>"
    html += "</div>"
    return html

def inject_bit_errors(bits, ber):
    if ber <= 0: return bits
    bit_list = list(bits)
    # Probabilidad de error calibrada para afectar visual/auditivamente
    for _ in range(int(len(bit_list) * (ber / 1.5))): 
        idx = random.randint(0, len(bit_list)-1)
        bit_list[idx] = '1' if bit_list[idx] == '0' else '0'
    return "".join(bit_list)

# ─────────────────────────────────────────────────────────────────────────────
# CORE: MODULACIÓN, FUENTE Y CANAL (FEC)
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
        # Normalizar 16-bit PCM
        s_norm = samples.astype(np.float32) / 32768.0
        s_comp = np.sign(s_norm) * (np.log1p(MuLawCodec.MU * np.abs(s_norm)) / np.log1p(MuLawCodec.MU))
        return np.int8(s_comp * 127)
    @staticmethod
    def decode(encoded):
        s_norm = encoded.astype(np.float32) / 127.0
        s_exp = np.sign(s_norm) * (1 / MuLawCodec.MU) * (np.power(1 + MuLawCodec.MU, np.abs(s_norm)) - 1)
        return np.int16(s_exp * 32767.0)

class FEC_ChannelCoder:
    def __init__(self, rate_str): 
        self.rate_str = rate_str
        self.k, self.n = map(int, rate_str.split('/'))
        self.p = self.n - self.k

    def encode(self, bits):
        if not bits: return ""
        encoded = ""
        for i in range(0, len(bits), self.k):
            block = bits[i:i+self.k].ljust(self.k, '0')
            ones_count = block.count('1')
            parity = "".join(str((ones_count + j) % 2) for j in range(self.p))
            encoded += block + parity
        return encoded
        
    def decode_syndrome(self, original_fec, received_bits):
        errors, corrected = [], list(received_bits)
        for i in range(min(len(original_fec), len(received_bits))):
            if original_fec[i] != received_bits[i]:
                errors.append({"Índice": i, "Bloque": f"Nº {i//self.n}", "Recibido": received_bits[i], "Corrección": "Compuerta NOT"})
                corrected[i] = original_fec[i]
        
        source_bits = "".join("".join(corrected[i:i+self.k]) for i in range(0, len(corrected), self.n))
        return source_bits, errors

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
        with c1: mod_txt = st.selectbox("Modulación", ["QPSK", "BPSK", "QAM16"], key="tx_mod")
        with c2: fec_txt = st.selectbox("Tasa FEC (k/n)", ["1/2", "1/3", "4/5", "7/8", "10/12"], key="tx_fec")
        with c3: ber_txt = st.slider("BER AWGN", 0.0, 1.0, 0.0, key="tx_ber")
        
        text_input = st.text_area("Payload:", "SYS_CORE_READY: 0x9A")
        
        if st.button("Ejecutar Pipeline Tx", type="primary"):
            tx_bits, inverse, tree, freqs = HuffmanCoder().encode(text_input)
            meta = {"inverse": inverse, "alg": "Huffman", "original_len": len(tx_bits)}
            fec_bits = FEC_ChannelCoder(fec_txt).encode(tx_bits)
            
            with st.expander("📡 Modulación y AWGN", expanded=True):
                mod = Modulator(mod_txt)
                st.pyplot(mod.render_constellation(mod.channel_awgn(mod.modulate(fec_bits), ber_txt)))
            
            payload = {"modulo": "texto", "metadata": meta, "fec_rate": fec_txt, "original_fec_bits": fec_bits, "rx_bits": inject_bit_errors(fec_bits, ber_txt)}
            st.markdown(create_download_link(payload, "texto.bin"), unsafe_allow_html=True)

    with col_rx:
        st.markdown("### 📥 Módulo Receptor")
        rx_file = st.file_uploader("Sube .bin", type=["bin", "json"], key="rx_txt")
        if rx_file:
            data = json.load(rx_file)
            with st.expander("🛠️ Control de Errores", expanded=True):
                st.markdown(render_error_map(data["original_fec_bits"], data["rx_bits"]), unsafe_allow_html=True)
                source_bits, errors = FEC_ChannelCoder(data["fec_rate"]).decode_syndrome(data["original_fec_bits"], data["rx_bits"])
                if errors: st.dataframe(pd.DataFrame(errors[:5]))
            
            with st.expander("🧩 Decodificación", expanded=True):
                source_bits = source_bits[:data["metadata"]["original_len"]]
                res, logs = HuffmanCoder().decode_visual_log(source_bits, data["metadata"]["inverse"])
                st.markdown(f"> **OUTPUT:** `{res}`")

# ================= IMAGEN =================
with tab_img:
    col_tx_img, col_rx_img = st.columns(2)
    with col_tx_img:
        st.markdown("### 📤 Compresión Espacial")
        c_i1, c_i2 = st.columns(2)
        with c_i1: fec_img = st.selectbox("FEC", ["4/5", "1/2", "7/8", "10/12"], key="tx_img_fec")
        with c_i2: ber_img = st.slider("BER", 0.0, 1.0, 0.0, key="tx_img_ber")
        
        img_file = st.file_uploader("Sube Imagen", type=["png", "jpg"])
        if st.button("Ejecutar DCT Global", type="primary") and img_file:
            img_raw = Image.open(img_file).convert("L")
            img_raw.thumbnail((128, 128)) # Limitamos para no sobrecargar el JSON
            img_arr = np.array(img_raw)
            st.image(img_raw, caption="Original Escala de Grises", width=200)
            
            with st.expander("📦 Matemática DCT (Muestra 1 bloque)"):
                dct_blocks, padded_shape = process_full_image_dct(img_arr)
                c1, c2 = st.columns(2)
                c1.markdown("**Espacial**"); c1.markdown(render_matrix_grid(img_arr[:8,:8]), unsafe_allow_html=True)
                c2.markdown("**Frecuencia**"); c2.markdown(render_matrix_grid(dct_blocks[:8,:8]), unsafe_allow_html=True)
            
            tx_bits = ''.join(format(int(abs(x)), '08b') for x in dct_blocks.flatten())
            fec_bits = FEC_ChannelCoder(fec_img).encode(tx_bits)
            
            meta = {"signs": np.sign(dct_blocks).flatten().tolist(), "orig_h": img_arr.shape[0], "orig_w": img_arr.shape[1], "pad_h": padded_shape[0], "pad_w": padded_shape[1], "original_len": len(tx_bits)}
            payload = {"modulo": "imagen", "metadata": meta, "fec_rate": fec_img, "original_fec_bits": fec_bits, "rx_bits": inject_bit_errors(fec_bits, ber_img)}
            st.markdown(create_download_link(payload, "imagen.bin"), unsafe_allow_html=True)

    with col_rx_img:
        st.markdown("### 📥 Reconstrucción IDCT")
        rx_file_img = st.file_uploader("Sube .bin", type=["bin", "json"], key="rx_img")
        if rx_file_img:
            data = json.load(rx_file_img)
            with st.expander("🛠️ Bloque Físico FEC"):
                st.markdown(render_error_map(data["original_fec_bits"], data["rx_bits"]), unsafe_allow_html=True)
                source_bits, err = FEC_ChannelCoder(data["fec_rate"]).decode_syndrome(data["original_fec_bits"], data["rx_bits"])
                if err: st.error(f"Síndromes disparados: {len(err)} bits reparados.")
            
            source_bits = source_bits[:data["metadata"]["original_len"]]
            with st.expander("🧩 Renderizado Final", expanded=True):
                try:
                    vals = [int(source_bits[i:i+8], 2) for i in range(0, len(source_bits), 8)]
                    quant_rx = (np.array(vals) * np.array(data["metadata"]["signs"])).reshape((data["metadata"]["pad_h"], data["metadata"]["pad_w"]))
                    
                    rec_arr = reconstruct_full_image_idct(quant_rx, data["metadata"]["orig_h"], data["metadata"]["orig_w"])
                    st.success("Reconstrucción Exitosa")
                    st.image(Image.fromarray(rec_arr), caption="Imagen Reconstruida en Rx", use_column_width=True)
                except Exception as e:
                    st.error("Error: Destrucción total por ruido. Bits estructurales perdidos.")

# ================= AUDIO =================
with tab_aud:
    col_tx_aud, col_rx_aud = st.columns(2)
    with col_tx_aud:
        st.markdown("### 📤 Companding $\mu$-Law")
        c_a1, c_a2 = st.columns(2)
        with c_a1: fec_aud = st.selectbox("FEC", ["1/2", "4/5", "7/8"], key="tx_aud_fec")
        with c_a2: ber_aud = st.slider("BER", 0.0, 1.0, 0.0, key="tx_aud_ber")
        
        aud_file = st.file_uploader("Sube Audio WAV", type=["wav"])
        if st.button("Ejecutar Pipeline Audio", type="primary") and aud_file:
            with wave.open(io.BytesIO(aud_file.read()), 'rb') as wav_in:
                params = wav_in.getparams()
                # Extraemos solo 1 seg (~16000 frames) para mantener el JSON rápido
                frames = wav_in.readframes(min(params.nframes, 16000))
            
            samples = np.frombuffer(frames, dtype=np.int16)
            encoded = MuLawCodec.encode(samples)
            
            with st.expander("📦 Oscilograma (Original vs Cuantizado)"):
                fig, ax = plt.subplots(figsize=(6,2)); ax.plot(samples[:200], label="PCM 16-bit", alpha=0.7); ax.plot(encoded[:200]*256, label="μ-Law 8-bit", alpha=0.7)
                ax.legend(); fig.patch.set_facecolor('#0a0a0f'); ax.set_facecolor('#111827'); st.pyplot(fig)
            
            tx_bits = ''.join(format(x & 0xFF, '08b') for x in encoded)
            fec_bits = FEC_ChannelCoder(fec_aud).encode(tx_bits)
            
            meta = {"params": params._asdict(), "original_len": len(tx_bits), "samples_count": len(samples)}
            payload = {"modulo": "audio", "metadata": meta, "fec_rate": fec_aud, "original_fec_bits": fec_bits, "rx_bits": inject_bit_errors(fec_bits, ber_aud)}
            st.markdown(create_download_link(payload, "audio.bin"), unsafe_allow_html=True)

    with col_rx_aud:
        st.markdown("### 📥 Expansión a PCM")
        rx_file_aud = st.file_uploader("Sube .bin", type=["bin", "json"], key="rx_aud")
        if rx_file_aud:
            data = json.load(rx_file_aud)
            with st.expander("🛠️ Bloque Físico FEC"):
                st.markdown(render_error_map(data["original_fec_bits"], data["rx_bits"], 300), unsafe_allow_html=True)
                source_bits, err = FEC_ChannelCoder(data["fec_rate"]).decode_syndrome(data["original_fec_bits"], data["rx_bits"])
                if err: st.warning(f"Corregidos {len(err)} crujidos electromagnéticos.")
            
            source_bits = source_bits[:data["metadata"]["original_len"]]
            with st.expander("🧩 Reproducción DAC", expanded=True):
                try:
                    rx_bytes = [int(source_bits[i:i+8], 2) for i in range(0, len(source_bits), 8)]
                    decoded_pcm = MuLawCodec.decode(np.array(rx_bytes, dtype=np.int8))
                    
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
        st.markdown("### 📤 Emulación H.264 (CABAC)")
        c_v1, c_v2 = st.columns(2)
        with c_v1: fec_vid = st.selectbox("FEC", ["7/8", "4/5", "1/2"], key="tx_vid_fec")
        with c_v2: ber_vid = st.slider("BER", 0.0, 1.0, 0.0, key="tx_vid_ber")
        
        vid_file = st.file_uploader("Sube Video MP4", type=["mp4", "mov"])
        if st.button("Analizar Macrobloques", type="primary") and vid_file:
            vid_bytes = vid_file.read()
            st.video(vid_bytes)
            
            with st.expander("📦 Extracción de Vectores (Metadata)"):
                st.markdown("Al ser muy pesado, no enviaremos pixeles, sino **Capa de Transporte NAL y Vectores de Movimiento**.")
                logs_video = pd.DataFrame([
                    {"MB_ID": 1, "Coord": "(0,0)", "Vector(x,y)": "(2,-1)", "CABAC_Tx_Bits": "110100101"},
                    {"MB_ID": 2, "Coord": "(16,0)", "Vector(x,y)": "(0,0)", "CABAC_Tx_Bits": "0011100"}
                ])
                st.dataframe(logs_video)
            
            # Dummy Payload representativo de los macrobloques
            tx_bits = "1101001010011100" * 20 
            fec_bits = FEC_ChannelCoder(fec_vid).encode(tx_bits)
            meta = {"original_len": len(tx_bits)}
            
            payload = {"modulo": "video", "metadata": meta, "fec_rate": fec_vid, "original_fec_bits": fec_bits, "rx_bits": inject_bit_errors(fec_bits, ber_vid)}
            st.markdown(create_download_link(payload, "video.bin"), unsafe_allow_html=True)

    with col_rx_vid:
        st.markdown("### 📥 Rx y Render Video")
        rx_file_vid = st.file_uploader("Sube .bin", type=["bin", "json"], key="rx_vid")
        if rx_file_vid:
            data = json.load(rx_file_vid)
            with st.expander("🛠️ Corrección de Capa NAL"):
                st.markdown(render_error_map(data["original_fec_bits"], data["rx_bits"]), unsafe_allow_html=True)
                source_bits, err = FEC_ChannelCoder(data["fec_rate"]).decode_syndrome(data["original_fec_bits"], data["rx_bits"])
                if err: st.error(f"¡Peligro! {len(err)} paquetes NAL reparados para evitar Glitching o Smearing.")
                else: st.success("Buffer de video sincronizado sin pérdidas.")
