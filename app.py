import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
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

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG Y CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multimedia Telecom Lab", layout="wide", page_icon="📡")

st.markdown("""
<style>
    .stApp{background:#0a0a0f;color:#e0e0e0;}
    h1,h2,h3{color:#00ffff;}
    div[data-testid='metric-container']{background:#111827;border:1px solid #9d4edd;border-radius:12px;padding:12px;}
    code{color:#00ffff; background: #111827;}
    html,body,[class*='css']{font-family:'Courier New', monospace;}
    table{color:white;}
    .dl-btn { display: inline-block; background: linear-gradient(135deg, #9d4edd 0%, #7b2cbf 100%); color: white !important; padding: 0.55rem 2rem; border-radius: 10px; font-weight: 700; text-decoration: none; font-size: 0.85rem; box-shadow: 0 2px 12px rgba(157,78,221,0.2); transition: all 0.2s ease; margin-top: 10px; text-align: center; width: 100%;}
    .dl-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(157,78,221,0.4); text-decoration: none; }
</style>
""", unsafe_allow_html=True)

st.title("📡 Multimedia Transmission Simulator")
st.caption("Pipeline completo: Fuente → Codificación de Canal (FEC) → Modulación → AWGN → Recepción")

# ─────────────────────────────────────────────────────────────────────────────
#  DATA CLASSES Y CORE
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class InformationMetrics:
    entropy: float
    avg_length: float
    efficiency: float
    redundancy: float

class InformationTheory:
    @staticmethod
    def calculate_entropy(data_bytes):
        counter = Counter(data_bytes)
        total = len(data_bytes)
        probs = [v / total for v in counter.values()]
        entropy = -sum(p * np.log2(p) for p in probs)
        return entropy, counter

    @staticmethod
    def metrics(data_bytes):
        entropy, _ = InformationTheory.calculate_entropy(data_bytes)
        avg_length = 8
        efficiency = (entropy / avg_length) if avg_length > 0 else 0
        redundancy = 1 - efficiency
        return InformationMetrics(entropy, avg_length, efficiency, redundancy)

class SourceCoder(ABC):
    @abstractmethod
    def encode(self, data): pass
    @abstractmethod
    def decode(self, data, *args): pass

@dataclass(order=True)
class HuffmanNode:
    freq: int
    symbol: any = field(compare=False)
    left: any = field(compare=False, default=None)
    right: any = field(compare=False, default=None)

class HuffmanCoder(SourceCoder):
    def build_tree(self, text):
        freq = Counter(text)
        heap = [HuffmanNode(v, k) for k, v in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(left.freq + right.freq, None, left, right)
            heapq.heappush(heap, merged)
        return heap[0] if heap else None, freq

    def generate_codes(self, node, current='', codes=None):
        if codes is None: codes = {}
        if not node: return codes
        if node.symbol is not None: codes[node.symbol] = current if current else "0"
        if node.left: self.generate_codes(node.left, current + '0', codes)
        if node.right: self.generate_codes(node.right, current + '1', codes)
        return codes

    def encode(self, text):
        tree, freq = self.build_tree(text)
        codes = self.generate_codes(tree)
        encoded = ''.join(codes[ch] for ch in text)
        inverse = {v: k for k, v in codes.items()}
        return encoded, codes, inverse, tree, freq

    def decode(self, encoded, inverse):
        current = ''
        decoded = []
        for bit in encoded:
            current += bit
            if current in inverse:
                decoded.append(inverse[current])
                current = ''
        if isinstance(list(inverse.values())[0], int):
            return bytes(decoded)
        return ''.join(decoded)

class LZWCoder(SourceCoder):
    def encode(self, text):
        dictionary = {chr(i): i for i in range(256)}
        string = ''
        code = 256
        result = []
        logs = []
        for symbol in text:
            temp = string + symbol
            if temp in dictionary:
                string = temp
            else:
                result.append(dictionary[string])
                dictionary[temp] = code
                logs.append({"entry": temp, "code": code})
                code += 1
                string = symbol
        if string:
            result.append(dictionary[string])
        return result, dictionary, logs

    def decode(self, compressed, *args):
        dictionary = {i: chr(i) for i in range(256)}
        if not compressed: return ""
        result = ""
        prev = compressed[0]
        result += dictionary[prev]
        code = 256
        for current in compressed[1:]:
            if current in dictionary:
                entry = dictionary[current]
            elif current == code:
                entry = dictionary[prev] + dictionary[prev][0]
            else:
                break
            result += entry
            dictionary[code] = dictionary[prev] + entry[0]
            code += 1
            prev = current
        return result

class RLECoder(SourceCoder):
    def encode(self, text):
        if not text: return []
        encoded = []
        i = 0
        while i < len(text):
            count = 1
            while i + 1 < len(text) and text[i] == text[i + 1] and count < 255:
                count += 1
                i += 1
            encoded.append((text[i], count))
            i += 1
        return encoded

    def decode(self, data, *args):
        if not data: return ""
        if isinstance(data[0][0], int):
            return bytes(sum(([ch] * count for ch, count in data), []))
        return ''.join(ch * count for ch, count in data)

# ─────────────────────────────────────────────────────────────────────────────
#  TELECOM PIPELINE (CHANNEL & MODULATION)
# ─────────────────────────────────────────────────────────────────────────────
class ChannelCoder:
    def __init__(self, rate='1/2'):
        self.rate = rate

    def encode(self, bits):
        # Implementación simplificada del control de error por paridad
        if not bits: return ""
        parity = ''.join('1' if bits[i:i+2].count('1') % 2 else '0' for i in range(0, len(bits), 2))
        return bits + parity

    def syndrome(self, original, received):
        errors = []
        corrected = list(received)
        for i in range(min(len(original), len(received))):
            if original[i] != received[i]:
                errors.append({"Bit Index": i, "Original": original[i], "Recibido": received[i], "Acción": "Corregido"})
                corrected[i] = original[i]
        return ''.join(corrected), errors

class Modulator:
    def __init__(self, scheme='BPSK'):
        self.scheme = scheme

    def modulate(self, bits):
        if not bits: return np.array([])
        if self.scheme == 'BPSK':
            return np.array([1 if b == '1' else -1 for b in bits])
        elif self.scheme == 'QPSK':
            symbols = []
            for i in range(0, len(bits), 2):
                pair = bits[i:i+2].ljust(2, '0')
                mapping = {'00': (-1, -1), '01': (-1, 1), '10': (1, -1), '11': (1, 1)}
                symbols.append(mapping[pair])
            return np.array(symbols)
        elif self.scheme == 'QAM16':
            levels = [-3, -1, 1, 3]
            symbols = []
            for i in range(0, len(bits), 4):
                chunk = bits[i:i+4].ljust(4, '0')
                i_val = levels[int(chunk[:2], 2)]
                q_val = levels[int(chunk[2:], 2)]
                symbols.append((i_val, q_val))
            return np.array(symbols)

    def add_awgn(self, signal, noise_level):
        if len(signal) == 0: return signal
        noise = np.random.normal(0, noise_level, signal.shape)
        return signal + noise

# ─────────────────────────────────────────────────────────────────────────────
#  MULTIMEDIA PROCESSORS
# ─────────────────────────────────────────────────────────────────────────────
class DCTProcessor:
    @staticmethod
    def process(image_array):
        gray = image_array.convert('L')
        img_np = np.array(gray)
        block = img_np[:8, :8] if img_np.shape[0] >= 8 and img_np.shape[1] >= 8 else np.zeros((8,8))
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        quant = np.round(dct_block / 10)
        recovered = idct(idct((quant * 10).T, norm='ortho').T, norm='ortho')
        return block, dct_block, quant, recovered

class ADPCMCodec:
    def encode(self, samples):
        prev = 0
        encoded = []
        logs = []
        for i, s in enumerate(samples):
            diff = int(s) - prev
            q = int(diff / 256)
            prev += q * 256
            encoded.append(q & 0xFF)
            if i < 15:
                logs.append({"Sample Real": int(s), "Diferencia": diff, "Cuantizado": q, "Reconstruido": prev})
        return np.array(encoded, dtype=np.uint8), logs

    def decode(self, encoded):
        prev = 0
        decoded = []
        for q in encoded:
            q_signed = np.int8(q)
            prev += int(q_signed) * 256
            decoded.append(prev)
        return np.array(decoded, dtype=np.int16)

class MuLawCodec:
    MU = 255
    @staticmethod
    def encode(samples):
        samples = samples.astype(np.float32)
        max_val = np.max(np.abs(samples)) if np.max(np.abs(samples)) > 0 else 1
        normalized = samples / max_val
        encoded = np.sign(normalized) * np.log1p(MuLawCodec.MU * np.abs(normalized)) / np.log1p(MuLawCodec.MU)
        return ((encoded + 1) / 2 * 255).astype(np.uint8)

class VideoSimulator:
    @staticmethod
    def generate_logs():
        return pd.DataFrame([
            {"Macroblock": 1, "Vect. Movimiento": "(2,1)", "Residual IDCT": 12, "Filtro": "Aplicado"},
            {"Macroblock": 2, "Vect. Movimiento": "(-1,0)", "Residual IDCT": 8, "Filtro": "Aplicado"},
            {"Macroblock": 3, "Vect. Movimiento": "(0,3)", "Residual IDCT": 4, "Filtro": "Aplicado"}
        ])

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS & UI
# ─────────────────────────────────────────────────────────────────────────────
def bits_from_bytes(data): return ''.join(format(byte, '08b') for byte in data)
def bytes_from_bits(bits): return bytes([int(bits[i:i+8].ljust(8, '0'), 2) for i in range(0, len(bits), 8)])

def create_download_link(data_bytes, filename):
    b64 = base64.b64encode(data_bytes).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" class="dl-btn">⬇️ Descargar Payload Modulado (.bin)</a>'

def show_metrics(metrics):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entropía (Shannon)", f"{metrics.entropy:.4f}")
    c2.metric("Longitud Promedio", f"{metrics.avg_length:.4f}")
    c3.metric("Eficiencia", f"{metrics.efficiency*100:.2f}%")
    c4.metric("Redundancia", f"{metrics.redundancy*100:.2f}%")

def render_pipeline_canal(tx_bits_sample, modulo, canal, ber, identifier):
    st.markdown("---")
    st.subheader("📡 Enlace de Telecomunicaciones (Tx/Rx)")
    
    # 1. CODIFICACIÓN DE CANAL
    fec_bits = canal.encode(tx_bits_sample)
    
    # 2. MODULACIÓN Y AWGN
    signal = modulo.modulate(fec_bits)
    noisy = modulo.add_awgn(signal, ber)
    
    # 3. RECEPCIÓN (Simulación de ruido flip bits)
    rx_bits_list = list(fec_bits)
    if ber > 0 and len(rx_bits_list) > 0:
        # Inyectar error proporcional al BER
        err_idx = random.randint(0, len(rx_bits_list)-1)
        rx_bits_list[err_idx] = '1' if rx_bits_list[err_idx] == '0' else '0'
    rx_bits = "".join(rx_bits_list)
    
    # 4. DECODIFICACIÓN Y CONTROL DE ERROR
    corrected_bits, errors = canal.syndrome(fec_bits, rx_bits)
    
    c1, c2 = st.columns([1.5, 2])
    with c1:
        st.markdown(f"**Diagrama de Constelación ({modulo.scheme})**")
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor('#0a0a0f')
        ax.set_facecolor('#111827')
        ax.tick_params(colors='#8b9cb5')
        for spine in ax.spines.values(): spine.set_edgecolor('#374151')
        
        if len(noisy) > 0:
            if modulo.scheme == 'BPSK':
                ax.scatter(noisy, np.zeros(len(noisy)), color='#00ffff', alpha=0.6, s=10)
            else:
                ax.scatter(noisy[:, 0], noisy[:, 1], color='#9d4edd', alpha=0.6, s=10)
        st.pyplot(fig)

    with c2:
        st.markdown("**Control de Errores (Síndrome)**")
        if not errors:
            st.success("Transmisión limpia. El decodificador no detectó alteraciones por el ruido.")
        else:
            st.warning("Se detectaron y corrigieron errores en la trama:")
            st.dataframe(pd.DataFrame(errors), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TABS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────
text_tab, image_tab, audio_tab, video_tab = st.tabs(["📝 Texto", "🖼 Imagen", "🎵 Audio", "🎬 Video"])

# ---------- TAB TEXTO ----------
with text_tab:
    st.header("📄 Codificación y Transmisión de Texto")
    text_input = st.text_area("Ingrese texto para transmitir", "Sistemas de telecomunicaciones avanzados.")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: algorithm = st.selectbox("1. Fuente", ["Huffman", "LZW", "RLE"], key="t_src")
    with c2: rate = st.selectbox("2. Canal (FEC)", ["1/2", "2/3", "3/4"], key="t_chn")
    with c3: modulation = st.selectbox("3. Modulación", ["BPSK", "QPSK", "QAM16"], key="t_mod")
    with c4: ber = st.slider("4. Ruido AWGN", 0.0, 1.0, 0.1, key="t_ber")

    if st.button("Transmitir Pipeline de Texto", type="primary"):
        st.session_state.run_text = True

    if st.session_state.get("run_text", False) and text_input:
        raw_bytes = text_input.encode('utf-8')
        show_metrics(InformationTheory.metrics(raw_bytes))
        
        # FUENTE
        st.subheader(f"📦 Codificador de Fuente ({algorithm})")
        if algorithm == "Huffman":
            coder = HuffmanCoder()
            encoded, codes, inverse, tree, freq = coder.encode(text_input)
            decoded = coder.decode(encoded, inverse)
            tx_bits = encoded
            
            col1, col2 = st.columns(2)
            with col1: st.dataframe(pd.DataFrame(freq.items(), columns=["Símbolo", "Frecuencia"]), height=200)
            with col2: st.json(inverse, expanded=False)
            
        elif algorithm == "LZW":
            coder = LZWCoder()
            compressed, dictionary, logs = coder.encode(text_input)
            decoded = coder.decode(compressed)
            tx_bits = ''.join(format(x, '016b') for x in compressed)
            st.dataframe(pd.DataFrame(logs), height=200)
            
        else: # RLE
            coder = RLECoder()
            encoded = coder.encode(text_input)
            decoded = coder.decode(encoded)
            tx_bits = ''.join(format(ord(ch), '08b') + format(count, '08b') for ch, count in encoded)
            st.dataframe(pd.DataFrame(encoded, columns=["Símbolo", "Repeticiones"]), height=200)

        st.markdown(create_download_link(bytes_from_bits(tx_bits), 'payload_texto.bin'), unsafe_allow_html=True)
        
        # CANAL Y MODULACIÓN
        render_pipeline_canal(tx_bits[:1000], Modulator(modulation), ChannelCoder(rate), ber, "texto")
        
        st.markdown("---")
        st.success(f"**Texto Recuperado en el Receptor:**\n\n{decoded}")

# ---------- TAB IMAGEN ----------
with image_tab:
    st.header("🖼 Procesamiento y Transmisión de Imagen")
    image_file = st.file_uploader("Suba imagen", type=['png', 'jpg', 'jpeg'])
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: algorithm_img = st.selectbox("1. Fuente", ["DCT — JPEG-like", "Huffman", "RLE"], key="i_src")
    with c2: rate_img = st.selectbox("2. Canal (FEC)", ["1/2", "2/3", "3/4"], key="i_chn")
    with c3: modulation_img = st.selectbox("3. Modulación", ["BPSK", "QPSK", "QAM16"], key="i_mod")
    with c4: ber_img = st.slider("4. Ruido AWGN", 0.0, 1.0, 0.1, key="i_ber")

    if st.button("Transmitir Pipeline de Imagen", type="primary"):
        st.session_state.run_img = True

    if st.session_state.get("run_img", False) and image_file:
        image = Image.open(image_file)
        img_bytes = image_file.read()
        show_metrics(InformationTheory.metrics(img_bytes))
        
        st.subheader(f"📦 Codificador de Fuente ({algorithm_img})")
        if algorithm_img == "DCT — JPEG-like":
            block, dct_block, quant, recovered = DCTProcessor.process(image)
            c1, c2 = st.columns(2)
            c1.markdown("Matriz Cuantizada (Transmisión)")
            c1.dataframe(pd.DataFrame(quant))
            c2.markdown("Matriz IDCT (Reconstrucción)")
            c2.dataframe(pd.DataFrame(np.round(recovered, 2)))
            tx_bits = ''.join(format(int(b), '08b') for b in np.nditer(np.abs(quant)) if b < 256)
        elif algorithm_img == "Huffman":
            coder = HuffmanCoder()
            encoded, codes, inverse, tree, freq = coder.encode(list(img_bytes[:5000])) # Límite para no congelar
            tx_bits = encoded
            st.json({str(k): v for k, v in list(codes.items())[:20]}, expanded=False)
        else:
            coder = RLECoder()
            encoded = coder.encode(list(img_bytes[:5000]))
            tx_bits = ''.join(format(ch, '08b') + format(count, '08b') for ch, count in encoded)
            st.dataframe(pd.DataFrame(encoded[:20], columns=["Byte", "Repeticiones"]))
        
        st.markdown(create_download_link(bytes_from_bits(tx_bits[:5000]), 'payload_imagen.bin'), unsafe_allow_html=True)
        
        # CANAL Y MODULACIÓN
        render_pipeline_canal(tx_bits[:1000], Modulator(modulation_img), ChannelCoder(rate_img), ber_img, "img")
        
        st.markdown("---")
        st.image(image, caption="Imagen original y procesada a través del canal.")

# ---------- TAB AUDIO ----------
with audio_tab:
    st.header("🎵 Procesamiento y Transmisión de Audio")
    audio_file = st.file_uploader("Suba audio WAV", type=['wav'])
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: algorithm_aud = st.selectbox("1. Fuente", ["ADPCM", "μ-Law G.711"], key="a_src")
    with c2: rate_aud = st.selectbox("2. Canal (FEC)", ["1/2", "2/3", "3/4"], key="a_chn")
    with c3: modulation_aud = st.selectbox("3. Modulación", ["BPSK", "QPSK", "QAM16"], key="a_mod")
    with c4: ber_aud = st.slider("4. Ruido AWGN", 0.0, 1.0, 0.1, key="a_ber")

    if st.button("Transmitir Pipeline de Audio", type="primary"):
        st.session_state.run_aud = True

    if st.session_state.get("run_aud", False) and audio_file:
        wav_buffer = io.BytesIO(audio_file.read())
        with wave.open(wav_buffer, 'rb') as wav:
            params = wav.getparams()
            frames = wav.readframes(params.nframes)
        samples = np.frombuffer(frames, dtype=np.int16)
        
        show_metrics(InformationTheory.metrics(frames[:10000]))
        
        st.subheader(f"📦 Codificador de Fuente ({algorithm_aud})")
        if algorithm_aud == "ADPCM":
            codec = ADPCMCodec()
            encoded, logs = codec.encode(samples[:5000])
            st.dataframe(pd.DataFrame(logs), height=200)
            tx_bits = ''.join(format(q, '08b') for q in encoded)
        else:
            encoded = MuLawCodec.encode(samples[:5000])
            st.dataframe(pd.DataFrame({"Original 16-bit": samples[:20], "μ-Law 8-bit": encoded[:20]}))
            tx_bits = ''.join(format(q, '08b') for q in encoded)
            
        st.markdown(create_download_link(bytes_from_bits(tx_bits), 'payload_audio.bin'), unsafe_allow_html=True)
        
        # CANAL Y MODULACIÓN
        render_pipeline_canal(tx_bits[:1000], Modulator(modulation_aud), ChannelCoder(rate_aud), ber_aud, "aud")

# ---------- TAB VIDEO ----------
with video_tab:
    st.header("🎬 Simulación H.264 y Transmisión")
    video_file = st.file_uploader("Suba video", type=['mp4', 'mov', 'avi'])
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.selectbox("1. Fuente", ["H.264 / HEVC (Simulado)"], disabled=True)
    with c2: rate_vid = st.selectbox("2. Canal (FEC)", ["1/2", "2/3", "3/4"], key="v_chn")
    with c3: modulation_vid = st.selectbox("3. Modulación", ["BPSK", "QPSK", "QAM16"], key="v_mod")
    with c4: ber_vid = st.slider("4. Ruido AWGN", 0.0, 1.0, 0.1, key="v_ber")

    if st.button("Transmitir Pipeline de Video", type="primary"):
        st.session_state.run_vid = True

    if st.session_state.get("run_vid", False) and video_file:
        video_bytes = video_file.read()
        show_metrics(InformationTheory.metrics(video_bytes[:50000]))
        
        st.subheader("📦 Codificador de Fuente (H.264 Simulado)")
        logs = VideoSimulator.generate_logs()
        st.dataframe(logs, use_container_width=True)
        
        # Simulamos unos bits basados en el video
        tx_bits = ''.join(format(b, '08b') for b in video_bytes[:100])
        st.markdown(create_download_link(bytes_from_bits(tx_bits), 'payload_video.bin'), unsafe_allow_html=True)
        
        # CANAL Y MODULACIÓN
        render_pipeline_canal(tx_bits, Modulator(modulation_vid), ChannelCoder(rate_vid), ber_vid, "vid")
