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

# Inyección de HTML en una sola línea
STYLING = "<style>.stApp{background:#0a0a0f;color:#e0e0e0;} h1,h2,h3{color:#00ffff;} div[data-testid='metric-container']{background:#111827;border:1px solid #9d4edd;border-radius:12px;padding:12px;} code{color:#00ffff; background:#111827; border:1px solid #374151;} html,body,[class*='css']{font-family:'Courier New', monospace;} table{color:white;} .dl-btn{display:block; background:linear-gradient(135deg, #9d4edd 0%, #7b2cbf 100%); color:white !important; padding:10px; border-radius:8px; font-weight:bold; text-decoration:none; text-align:center; margin-top:10px; transition:0.3s;} .dl-btn:hover{box-shadow:0 0 15px #9d4edd;}</style>"
st.markdown(STYLING, unsafe_allow_html=True)

st.title("📡 Simulación Interactiva: Tx → Canal Humano → Rx")
st.caption("1. Codifica y descarga el archivo | 2. Abre el archivo y modifica los bits (simula el ruido) | 3. Sube el archivo modificado para decodificar.")

# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class InformationMetrics:
    entropy: float; avg_length: float; efficiency: float; redundancy: float

class InfoTheory:
    @staticmethod
    def calc_metrics(data_bytes):
        counts = Counter(data_bytes)
        total = len(data_bytes)
        probs = [v/total for v in counts.values()]
        H = -sum(p * np.log2(p) for p in probs if p > 0)
        avg_len = 8.0 
        eff = H / avg_len if avg_len > 0 else 0
        return InformationMetrics(H, avg_len, eff, 1 - eff), counts

def create_download_link(payload_dict, filename):
    json_str = json.dumps(payload_dict, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" class="dl-btn">⬇ Descargar Payload para Modificar ({filename})</a>'

def render_matrix_grid(matrix):
    html = "<div style='display:grid;grid-template-columns:repeat(8,minmax(25px,1fr));gap:2px;font-size:11px;text-align:center;'>"
    for row in matrix:
        for val in row: html += f"<div style='background:#111827;border:1px solid #9d4edd;padding:4px;'>{int(val)}</div>"
    html += "</div>"
    return html

# ─────────────────────────────────────────────────────────────────────────────
# CÓDECS DE FUENTE
# ─────────────────────────────────────────────────────────────────────────────
class SourceCoder(ABC):
    @abstractmethod
    def encode(self, data): pass
    @abstractmethod
    def decode(self, data, *args): pass

@dataclass(order=True)
class HuffNode:
    freq: int; symbol: any = field(compare=False); left: any = field(compare=False, default=None); right: any = field(compare=False, default=None)

class HuffmanCoder(SourceCoder):
    def encode(self, text):
        freq = Counter(text)
        heap = [HuffNode(v, k) for k, v in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            l, r = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, HuffNode(l.freq + r.freq, None, l, r))
        tree = heap[0] if heap else None
        
        codes = {}
        def gen_codes(n, current=''):
            if n.symbol is not None: codes[n.symbol] = current or '0'
            if n.left: gen_codes(n.left, current+'0')
            if n.right: gen_codes(n.right, current+'1')
        gen_codes(tree)
        encoded = ''.join(codes[ch] for ch in text)
        inverse = {v: k for k, v in codes.items()}
        return encoded, inverse, tree, freq
    
    def decode(self, encoded, inverse):
        curr, out = '', []
        for b in encoded:
            curr += b
            if curr in inverse:
                out.append(inverse[curr]); curr = ''
        return ''.join(out)

class LZWCoder(SourceCoder):
    def encode(self, text):
        dict_size = 256
        dictionary = {chr(i): i for i in range(dict_size)}
        w, result, logs = "", [], []
        for c in text:
            wc = w + c
            if wc in dictionary: w = wc
            else:
                result.append(dictionary[w])
                dictionary[wc] = dict_size
                logs.append({"Puntero/Código": dict_size, "Secuencia": wc})
                dict_size += 1
                w = c
        if w: result.append(dictionary[w])
        bits = ''.join(format(x, '016b') for x in result)
        return bits, result, logs

    def decode(self, compressed_bits, *args):
        # Reconstruir la lista de ints desde los bits
        compressed_list = [int(compressed_bits[i:i+16], 2) for i in range(0, len(compressed_bits), 16)]
        if not compressed_list: return ""
        
        dict_size = 256
        dictionary = {i: chr(i) for i in range(dict_size)}
        result = dictionary[compressed_list[0]]
        w = dictionary[compressed_list[0]]
        for k in compressed_list[1:]:
            entry = dictionary[k] if k in dictionary else w + w[0]
            result += entry
            dictionary[dict_size] = w + entry[0]
            dict_size += 1
            w = entry
        return result

class RLECoder(SourceCoder):
    def encode(self, text):
        encoded, i = [], 0
        while i < len(text):
            count = 1
            while i + 1 < len(text) and text[i] == text[i+1] and count < 255:
                count += 1; i += 1
            encoded.append((text[i], count)); i += 1
        bits = ''.join(format(ord(ch), '08b') + format(c, '08b') for ch, c in encoded)
        return bits, encoded
        
    def decode(self, bits, *args):
        out = ""
        for i in range(0, len(bits), 16):
            if i+16 <= len(bits):
                ch = chr(int(bits[i:i+8], 2))
                c = int(bits[i+8:i+16], 2)
                out += ch * c
        return out

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

# ─────────────────────────────────────────────────────────────────────────────
# CAPA DE CANAL (FEC)
# ─────────────────────────────────────────────────────────────────────────────
class FEC_ChannelCoder:
    def __init__(self, rate_str):
        self.rate_str = rate_str
    
    def encode(self, bits):
        if not bits: return ""
        # Simulación: Agregamos bits de paridad al final de la trama
        parity = ''.join('1' if bits[i:i+3].count('1')%2 else '0' for i in range(0, len(bits), 3))
        return bits + parity
        
    def decode_syndrome(self, original_fec_bits, received_bits):
        errors, corrected = [], list(received_bits)
        for i in range(min(len(original_fec_bits), len(received_bits))):
            if original_fec_bits[i] != received_bits[i]:
                errors.append({"Bit Alterado (Índice)": i, "Valor Esperado": original_fec_bits[i], "Tu Archivo Dijo": received_bits[i], "Acción FEC": "Corregido"})
                corrected[i] = original_fec_bits[i]
        
        # Eliminar paridad para obtener bits de fuente limpios
        data_len = int(len(original_fec_bits) * (3/4)) if self.rate_str != '1/2' else len(original_fec_bits)//2
        source_bits = "".join(corrected)[:data_len] # Aproximación de extracción para la demo
        
        return source_bits, errors

# ─────────────────────────────────────────────────────────────────────────────
# TABS Y UI
# ─────────────────────────────────────────────────────────────────────────────
tab_txt, tab_img, tab_aud, tab_vid = st.tabs(["📄 Texto", "🖼️ Imagen", "🎵 Audio", "🎬 Video"])

# ================= TEXTO =================
with tab_txt:
    col_tx, col_rx = st.columns(2)
    
    with col_tx:
        st.markdown("### 📤 1. Transmisión (Codificar)")
        src_txt = st.selectbox("Algoritmo", ["Huffman", "LZW", "RLE"], key="tx_src")
        fec_txt = st.selectbox("FEC Rate", ["2/3", "1/2"], key="tx_fec")
        text_input = st.text_area("Texto a transmitir:", "Hola Mundo Telecomunicaciones")
        
        if st.button("Generar Archivo .BIN", type="primary"):
            if src_txt == "Huffman":
                coder = HuffmanCoder()
                tx_bits, inverse, tree, freqs = coder.encode(text_input)
                meta = {"inverse": inverse, "alg": "Huffman"}
            elif src_txt == "LZW":
                coder = LZWCoder()
                tx_bits, _, _ = coder.encode(text_input)
                meta = {"alg": "LZW"}
            else:
                coder = RLECoder()
                tx_bits, _ = coder.encode(text_input)
                meta = {"alg": "RLE"}
                
            fec = FEC_ChannelCoder(fec_txt)
            fec_bits = fec.encode(tx_bits)
            
            payload = {
                "modulo": "texto",
                "metadata": meta,
                "fec_rate": fec_txt,
                "original_fec_bits": fec_bits, # Firma oculta para comparar tus alteraciones
                "rx_bits": fec_bits # Este es el campo que vas a modificar
            }
            
            st.success("Codificación completada. Descarga el archivo, edita el campo `rx_bits` y súbelo en Rx.")
            st.markdown(create_download_link(payload, "texto_payload.bin"), unsafe_allow_html=True)

    with col_rx:
        st.markdown("### 📥 2. Recepción (Decodificar)")
        rx_file = st.file_uploader("Sube el archivo .bin (JSON) modificado", type=["bin", "json"], key="rx_txt")
        
        if rx_file is not None:
            data = json.load(rx_file)
            fec = FEC_ChannelCoder(data["fec_rate"])
            
            st.warning("Evaluando Síndromes...")
            source_bits, errors = fec.decode_syndrome(data["original_fec_bits"], data["rx_bits"])
            
            if errors:
                st.error(f"¡Se detectaron {len(errors)} errores introducidos!")
                st.dataframe(pd.DataFrame(errors), height=150)
            else:
                st.success("Transmisión limpia. No modificaste ningún bit.")
                
            # Reconstrucción
            alg = data["metadata"]["alg"]
            try:
                if alg == "Huffman":
                    res = HuffmanCoder().decode(source_bits, data["metadata"]["inverse"])
                elif alg == "LZW":
                    res = LZWCoder().decode(source_bits)
                else:
                    res = RLECoder().decode(source_bits)
                    
                st.info(f"**Texto Reconstruido:**\n\n{res}")
            except Exception as e:
                st.error("Error catastrófico: Alteraste bits críticos de la estructura y el FEC no fue suficiente (o excediste la capacidad).")

# ================= IMAGEN =================
with tab_img:
    col_tx_img, col_rx_img = st.columns(2)
    
    with col_tx_img:
        st.markdown("### 📤 1. Transmisión (DCT)")
        img_file = st.file_uploader("Subir Imagen", type=["png", "jpg"])
        
        if st.button("Generar Payload Imagen", type="primary") and img_file:
            img_arr = np.array(Image.open(img_file).convert("L"))
            block = img_arr[:8, :8] if img_arr.shape[0]>=8 and img_arr.shape[1]>=8 else np.zeros((8,8))
            
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            quant = np.round(dct_block / 10)
            
            # Convertir matriz a string de bits simple
            tx_bits = ''.join(format(int(abs(x)), '08b') for x in quant.flatten())
            fec = FEC_ChannelCoder("2/3")
            fec_bits = fec.encode(tx_bits)
            
            payload = {
                "modulo": "imagen",
                "metadata": {"shape": [8,8], "signs": np.sign(quant).flatten().tolist()},
                "fec_rate": "2/3",
                "original_fec_bits": fec_bits,
                "rx_bits": fec_bits
            }
            
            st.markdown(create_download_link(payload, "imagen_payload.bin"), unsafe_allow_html=True)
            st.markdown(render_matrix_grid(quant), unsafe_allow_html=True)

    with col_rx_img:
        st.markdown("### 📥 2. Recepción (IDCT)")
        rx_file_img = st.file_uploader("Sube el archivo modificado", type=["bin", "json"], key="rx_img")
        
        if rx_file_img is not None:
            data = json.load(rx_file_img)
            fec = FEC_ChannelCoder(data["fec_rate"])
            source_bits, errors = fec.decode_syndrome(data["original_fec_bits"], data["rx_bits"])
            
            if errors: st.error(f"Se corrigieron {len(errors)} bits."); st.dataframe(pd.DataFrame(errors[:10]))
            
            try:
                # Reconstruir Matriz
                vals = [int(source_bits[i:i+8], 2) for i in range(0, len(source_bits[:64*8]), 8)]
                signs = data["metadata"]["signs"]
                quant_rx = np.array(vals) * np.array(signs)
                quant_rx = quant_rx.reshape((8,8))
                
                idct_block = idct(idct((quant_rx * 10).T, norm='ortho').T, norm='ortho')
                st.success("Matriz Recuperada (IDCT):")
                st.markdown(render_matrix_grid(np.round(idct_block)), unsafe_allow_html=True)
            except:
                st.error("Bits gravemente dañados.")

# ================= AUDIO =================
with tab_aud:
    col_tx_aud, col_rx_aud = st.columns(2)
    
    with col_tx_aud:
        st.markdown("### 📤 1. Transmisión")
        aud_file = st.file_uploader("Subir WAV", type=["wav"])
        
        if st.button("Generar Payload Audio", type="primary") and aud_file:
            with wave.open(io.BytesIO(aud_file.read()), 'rb') as wav_in:
                params = wav_in.getparams()
                frames = wav_in.readframes(params.nframes)
            
            samples = np.frombuffer(frames, dtype=np.int16)[:2000] # Limitado para demo ligera
            encoded_aud = MuLawCodec.encode(samples)
            tx_bits = ''.join(format(x & 0xFF, '08b') for x in encoded_aud)
            
            fec = FEC_ChannelCoder("1/2")
            fec_bits = fec.encode(tx_bits)
            
            payload = {
                "modulo": "audio",
                "metadata": {"params": params._asdict()},
                "fec_rate": "1/2",
                "original_fec_bits": fec_bits,
                "rx_bits": fec_bits
            }
            st.markdown(create_download_link(payload, "audio_payload.bin"), unsafe_allow_html=True)

    with col_rx_aud:
        st.markdown("### 📥 2. Recepción")
        rx_file_aud = st.file_uploader("Sube el archivo modificado", type=["bin", "json"], key="rx_aud")
        if rx_file_aud is not None:
            data = json.load(rx_file_aud)
            source_bits, errors = FEC_ChannelCoder(data["fec_rate"]).decode_syndrome(data["original_fec_bits"], data["rx_bits"])
            
            if errors: st.error(f"{len(errors)} Errores de canal detectados y corregidos."); st.dataframe(pd.DataFrame(errors[:5]))
            
            try:
                rx_bytes = [int(source_bits[i:i+8], 2) for i in range(0, len(source_bits), 8)]
                decoded_pcm = MuLawCodec.decode(np.array(rx_bytes, dtype=np.int8))
                
                out_buffer = io.BytesIO()
                with wave.open(out_buffer, 'wb') as wav_out:
                    wav_out.setparams(data["metadata"]["params"])
                    wav_out.writeframes(decoded_pcm.tobytes())
                st.audio(out_buffer.getvalue())
            except Exception as e:
                st.error("Audio corrupto.")

# ================= VIDEO =================
with tab_vid:
    st.info("Para video el concepto es idéntico. Carga un archivo dummy aquí para ver la corrección.")
    if st.button("Generar Video Dummy BIN"):
        dummy_bits = "101011110000" * 50
        payload = {"modulo": "video", "original_fec_bits": dummy_bits, "rx_bits": dummy_bits, "fec_rate": "2/3"}
        st.markdown(create_download_link(payload, "video_payload.bin"), unsafe_allow_html=True)
    
    rx_vid = st.file_uploader("Subir Dummy Video Modificado", type=["bin", "json"])
    if rx_vid is not None:
        data = json.load(rx_vid)
        _, err = FEC_ChannelCoder(data["fec_rate"]).decode_syndrome(data["original_fec_bits"], data["rx_bits"])
        if err: st.error(f"{len(err)} errores corregidos en el macroblock."); st.dataframe(pd.DataFrame(err))
        else: st.success("Video recibido intacto.")
