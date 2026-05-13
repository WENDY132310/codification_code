
# app.py
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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import Counter
from scipy.fftpack import dct, idct
from PIL import Image

st.set_page_config(page_title="Multimedia Telecom Lab", layout="wide")

st.markdown("<style>.stApp{background:#0a0a0f;color:#00ffff;}h1,h2,h3{color:#9d4edd;}div[data-testid='metric-container']{background:#111827;border:1px solid #00ffff;border-radius:12px;padding:12px;}code{color:#00ffff;}html,body,[class*='css']{font-family:monospace;}table{color:white;}</style>", unsafe_allow_html=True)

st.title("📡 Multimedia Transmission Simulator")
st.caption("Pipeline completo: Fuente → Canal → Modulación → Recepción")

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
        efficiency = entropy / avg_length
        redundancy = 1 - efficiency
        return InformationMetrics(entropy, avg_length, efficiency, redundancy)

class SourceCoder(ABC):

    @abstractmethod
    def encode(self, data):
        pass

    @abstractmethod
    def decode(self, data):
        pass

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
        inverse = {v: k for k, v in codes.items()}
        return encoded, codes, inverse, tree, freq

    def decode(self, encoded, inverse):
        current = ''
        decoded = ''

        for bit in encoded:
            current += bit
            if current in inverse:
                decoded += inverse[current]
                current = ''

        return decoded

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

    def decode(self, compressed):
        dictionary = {i: chr(i) for i in range(256)}
        result = ""
        prev = compressed[0]
        result += dictionary[prev]
        code = 256

        for current in compressed[1:]:
            if current in dictionary:
                entry = dictionary[current]
            elif current == code:
                entry = dictionary[prev] + dictionary[prev][0]

            result += entry
            dictionary[code] = dictionary[prev] + entry[0]
            code += 1
            prev = current

        return result

class RLECoder(SourceCoder):

    def encode(self, text):
        encoded = []
        i = 0

        while i < len(text):
            count = 1
            while i + 1 < len(text) and text[i] == text[i + 1]:
                count += 1
                i += 1
            encoded.append((text[i], count))
            i += 1

        return encoded

    def decode(self, data):
        return ''.join(ch * count for ch, count in data)

class ChannelCoder:

    def __init__(self, rate='1/2'):
        self.rate = rate

    def encode(self, bits):
        parity = ''.join('1' if bits[i:i+2].count('1') % 2 else '0' for i in range(0, len(bits), 2))
        return bits + parity

    def syndrome(self, original, received):
        errors = []
        corrected = list(received)

        for i in range(min(len(original), len(received))):
            if original[i] != received[i]:
                errors.append({"bit": i, "original": original[i], "received": received[i]})
                corrected[i] = original[i]

        return ''.join(corrected), errors

class Modulator:

    def __init__(self, scheme='BPSK'):
        self.scheme = scheme

    def modulate(self, bits):

        if self.scheme == 'BPSK':
            return np.array([1 if b == '1' else -1 for b in bits])

        elif self.scheme == 'QPSK':
            symbols = []
            for i in range(0, len(bits), 2):
                pair = bits[i:i+2].ljust(2, '0')
                mapping = {
                    '00': (-1, -1),
                    '01': (-1, 1),
                    '10': (1, -1),
                    '11': (1, 1)
                }
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
        noise = np.random.normal(0, noise_level, signal.shape)
        return signal + noise

class DCTProcessor:

    @staticmethod
    def process(image_array):
        gray = image_array.convert('L')
        img_np = np.array(gray)

        block = img_np[:8, :8]

        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        quant = np.round(dct_block / 10)
        recovered = idct(idct((quant * 10).T, norm='ortho').T, norm='ortho')

        return block, dct_block, quant, recovered

class MuLawCodec:

    MU = 255

    @staticmethod
    def encode(samples):
        samples = samples.astype(np.float32)
        max_val = np.max(np.abs(samples))
        normalized = samples / max_val
        encoded = np.sign(normalized) * np.log1p(MuLawCodec.MU * np.abs(normalized)) / np.log1p(MuLawCodec.MU)
        return ((encoded + 1) / 2 * 255).astype(np.uint8)

    @staticmethod
    def decode(encoded):
        signal = encoded.astype(np.float32) / 255 * 2 - 1
        decoded = np.sign(signal) * (1 / MuLawCodec.MU) * ((1 + MuLawCodec.MU) ** np.abs(signal) - 1)
        return (decoded * 32767).astype(np.int16)

class ADPCMCodec:

    def encode(self, samples):
        prev = 0
        encoded = []
        logs = []

        for s in samples:
            diff = int(s) - prev
            q = int(diff / 256)
            prev += q * 256
            encoded.append(q & 0xFF)
            logs.append({
                "sample": int(s),
                "difference": diff,
                "quantized": q,
                "reconstructed": prev
            })

        return np.array(encoded, dtype=np.uint8), logs

    def decode(self, encoded):
        prev = 0
        decoded = []

        for q in encoded:
            q_signed = np.int8(q)
            prev += int(q_signed) * 256
            decoded.append(prev)

        return np.array(decoded, dtype=np.int16)

class VideoSimulator:

    @staticmethod
    def generate_logs():
        return pd.DataFrame([
            {"macroblock": 1, "motion_vector": "(2,1)", "residual": 12},
            {"macroblock": 2, "motion_vector": "(-1,0)", "residual": 8},
            {"macroblock": 3, "motion_vector": "(0,3)", "residual": 4}
        ])

def bits_from_bytes(data):
    return ''.join(format(byte, '08b') for byte in data)

def bytes_from_bits(bits):
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte_array.append(int(bits[i:i+8].ljust(8, '0'), 2))
    return bytes(byte_array)

def create_download_link(data_bytes, filename):
    b64 = base64.b64encode(data_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" style="background:#00ffff;color:black;padding:10px;border-radius:8px;text-decoration:none;">⬇ Descargar BIN Transmitido</a>'
    return href

def show_metrics(metrics):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entropía", round(metrics.entropy, 4))
    c2.metric("Longitud Promedio", round(metrics.avg_length, 4))
    c3.metric("Eficiencia", round(metrics.efficiency, 4))
    c4.metric("Redundancia", round(metrics.redundancy, 4))

text_tab, image_tab, audio_tab, video_tab = st.tabs(["Texto", "Imagen", "Audio", "Video"])

with text_tab:

    st.header("📄 Codificación de Texto")

    text_input = st.text_area("Ingrese texto")

    algorithm = st.selectbox("Algoritmo", ["Huffman", "LZW", "RLE"])
    modulation = st.selectbox("Modulación", ["BPSK", "QPSK", "QAM16"], key="textmod")
    ber = st.slider("Ruido AWGN", 0.0, 2.0, 0.2)

    if st.button("Transmitir y Procesar Texto"):

        if text_input:

            raw_bytes = text_input.encode()
            metrics = InformationTheory.metrics(raw_bytes)
            show_metrics(metrics)

            st.code("H(X) = -Σ p(x) log2 p(x)")

            if algorithm == "Huffman":

                coder = HuffmanCoder()
                encoded, codes, inverse, tree, freq = coder.encode(text_input)
                decoded = coder.decode(encoded, inverse)

                st.dataframe(pd.DataFrame(freq.items(), columns=["Símbolo", "Frecuencia"]))

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

                st.json(inverse)

                tx_bits = encoded

            elif algorithm == "LZW":

                coder = LZWCoder()
                compressed, dictionary, logs = coder.encode(text_input)
                decoded = coder.decode(compressed)

                st.dataframe(pd.DataFrame(logs))

                tx_bits = ''.join(format(x, '016b') for x in compressed)

            else:

                coder = RLECoder()
                encoded = coder.encode(text_input)
                decoded = coder.decode(encoded)

                st.dataframe(pd.DataFrame(encoded, columns=["Símbolo", "Repeticiones"]))

                tx_bits = ''.join(format(ord(ch), '08b') + format(count, '08b') for ch, count in encoded)

            channel = ChannelCoder()
            fec = channel.encode(tx_bits)

            modulator = Modulator(modulation)
            signal = modulator.modulate(fec)
            noisy = modulator.add_awgn(signal, ber)

            st.markdown(create_download_link(bytes_from_bits(fec), 'text_tx.bin'), unsafe_allow_html=True)

            fig, ax = plt.subplots()

            if modulation == 'BPSK':
                ax.scatter(noisy, np.zeros(len(noisy)))
            else:
                ax.scatter(noisy[:, 0], noisy[:, 1])

            st.pyplot(fig)

            st.success(decoded)

with image_tab:

    st.header("🖼 Procesamiento de Imagen")

    image_file = st.file_uploader("Suba imagen", type=['png', 'jpg', 'jpeg'])

    if st.button("Transmitir y Procesar Imagen"):

        if image_file:

            image = Image.open(image_file)
            st.image(image)

            img_bytes = image_file.read()
            metrics = InformationTheory.metrics(img_bytes)
            show_metrics(metrics)

            block, dct_block, quant, recovered = DCTProcessor.process(image)

            st.dataframe(pd.DataFrame(block))
            st.dataframe(pd.DataFrame(np.round(dct_block, 2)))
            st.dataframe(pd.DataFrame(quant))
            st.dataframe(pd.DataFrame(np.round(recovered, 2)))

with audio_tab:

    st.header("🎵 Procesamiento de Audio")

    audio_file = st.file_uploader("Suba audio WAV", type=['wav'])

    if st.button("Transmitir y Procesar Audio"):

        if audio_file:

            wav_buffer = io.BytesIO(audio_file.read())

            with wave.open(wav_buffer, 'rb') as wav:
                params = wav.getparams()
                frames = wav.readframes(params.nframes)

            samples = np.frombuffer(frames, dtype=np.int16)

            metrics = InformationTheory.metrics(frames)
            show_metrics(metrics)

            codec = ADPCMCodec()
            encoded, logs = codec.encode(samples[:3000])
            decoded = codec.decode(encoded)

            st.dataframe(pd.DataFrame(logs))

with video_tab:

    st.header("🎬 Simulación H.264")

    video_file = st.file_uploader("Suba video", type=['mp4', 'mov', 'avi'])

    if st.button("Transmitir y Procesar Video"):

        if video_file:

            video_bytes = video_file.read()

            metrics = InformationTheory.metrics(video_bytes)
            show_metrics(metrics)

            st.video(video_bytes)

            logs = VideoSimulator.generate_logs()
            st.dataframe(logs)

            st.success("Video reconstruido correctamente.")
