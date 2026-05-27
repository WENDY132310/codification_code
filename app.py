from __future__ import annotations

import base64
import heapq
import io
import json
import math
import random
import struct
import time
import wave
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from scipy.fftpack import dct as scipy_dct, idct as scipy_idct
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ── Page Config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Telecom DSP Lab · Fuente & Canal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ═════════════════════════════════════════════════════════════════════════════
#  MERGED CSS — Terminal Lab Dark
# ═════════════════════════════════════════════════════════════════════════════
def inject_css() -> None:
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&display=swap');
:root{
    --bg-0:#030712;--bg-1:#070d1a;--bg-2:#0d1424;--bg-3:#141e30;--bg-4:#1a2540;
    --cyan:#06b6d4;--cyan-dim:rgba(6,182,212,0.12);
    --violet:#8b5cf6;--violet-dim:rgba(139,92,246,0.12);
    --green:#10b981;--amber:#f59e0b;--red:#ef4444;
    --txt:#cdd9e5;--txt-dim:#8b9cb5;--muted:#4e5f7a;
    --border:rgba(6,182,212,0.14);--border-strong:rgba(6,182,212,0.30);
    --radius-sm:8px;--radius-md:14px;--radius-lg:20px;
}
html,body,[class*="css"],[data-testid]{font-family:'Syne',sans-serif !important;background-color:var(--bg-0) !important;color:var(--txt) !important;}
*{box-sizing:border-box;}
#MainMenu,footer,header{visibility:hidden !important;}
.stDeployButton,[data-testid="stToolbar"]{display:none !important;}
section[data-testid="stSidebar"]{display:none !important;}
.main .block-container{padding:0 2rem 4rem !important;max-width:1440px !important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg-1);}
::-webkit-scrollbar-thumb{background:var(--cyan);border-radius:3px;}
/* HEADER */
.app-header-wrap{background:var(--bg-1);border-bottom:1px solid var(--border);padding:1.2rem 2rem 1.4rem;margin:0 -2rem 1.5rem;}
.app-header{display:flex;align-items:center;gap:1.25rem;}
.header-icon{width:52px;height:52px;background:linear-gradient(135deg,var(--cyan),var(--violet));border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;}
.app-title{font-size:1.6rem;font-weight:800;background:linear-gradient(100deg,var(--cyan) 0%,#a78bfa 60%,var(--violet) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-0.04em;margin:0;}
.app-subtitle{font-family:'IBM Plex Mono',monospace !important;color:var(--muted);font-size:0.68rem;letter-spacing:0.07em;margin-top:0.2rem;}
/* TABS */
.stTabs [data-baseweb="tab-list"]{background:var(--bg-2) !important;border-radius:var(--radius-md) !important;padding:5px !important;gap:4px !important;border:1px solid var(--border) !important;}
.stTabs [data-baseweb="tab"]{background:transparent !important;border-radius:var(--radius-sm) !important;color:var(--muted) !important;font-weight:600 !important;font-size:0.82rem !important;padding:0.55rem 1.4rem !important;border:none !important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,var(--cyan-dim),var(--violet-dim)) !important;color:var(--cyan) !important;border:1px solid var(--border-strong) !important;}
.stTabs [data-baseweb="tab-panel"]{padding:1.75rem 0 0 !important;}
/* SECTION LABELS */
.section-label{display:flex;align-items:center;gap:0.6rem;margin:1.5rem 0 0.85rem;}
.section-label .sl-icon{font-size:0.9rem;}
.section-label .sl-text{font-family:'IBM Plex Mono',monospace !important;font-size:0.65rem;font-weight:600;letter-spacing:0.18em;text-transform:uppercase;color:var(--cyan);}
.section-label::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border) 0%,transparent 100%);}
/* METRICS */
[data-testid="stMetric"]{background:var(--bg-2) !important;border:1px solid var(--border) !important;border-radius:var(--radius-md) !important;padding:1.1rem 1.3rem !important;position:relative;overflow:hidden;}
[data-testid="stMetric"]::before{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--violet),var(--cyan));}
[data-testid="stMetricLabel"]>div{font-family:'IBM Plex Mono',monospace !important;font-size:0.62rem !important;color:var(--muted) !important;letter-spacing:0.1em !important;text-transform:uppercase !important;}
[data-testid="stMetricValue"]>div{font-size:1.7rem !important;font-weight:700 !important;color:var(--cyan) !important;font-family:'IBM Plex Mono',monospace !important;letter-spacing:-0.03em !important;}
/* UPLOADER & BUTTONS */
[data-testid="stFileUploader"]{background:var(--bg-2) !important;border:1.5px dashed rgba(6,182,212,0.28) !important;border-radius:var(--radius-lg) !important;padding:1.5rem !important;transition:border-color 0.25s !important;}
[data-testid="stFileUploader"]:hover{border-color:var(--cyan) !important;}
.stButton>button{background:linear-gradient(135deg,var(--cyan) 0%,var(--violet) 100%) !important;color:#fff !important;border:none !important;border-radius:10px !important;font-weight:700 !important;font-family:'Syne',sans-serif !important;font-size:0.85rem !important;padding:0.55rem 2rem !important;letter-spacing:0.03em !important;transition:all 0.2s ease !important;box-shadow:0 2px 12px rgba(6,182,212,0.2) !important;}
.stButton>button:hover{transform:translateY(-2px) !important;box-shadow:0 6px 24px rgba(6,182,212,0.35) !important;}
/* INFO BOXES */
.info-box{display:flex;gap:0.9rem;align-items:flex-start;background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.22);border-radius:var(--radius-md);padding:1rem 1.25rem;margin:0.75rem 0;}
.info-box.amber{background:rgba(245,158,11,0.06);border-color:rgba(245,158,11,0.22);}
.info-box.green{background:rgba(16,185,129,0.06);border-color:rgba(16,185,129,0.22);}
.info-box.violet{background:rgba(139,92,246,0.06);border-color:rgba(139,92,246,0.22);}
.info-box .ib-content{font-family:'IBM Plex Mono',monospace !important;font-size:0.75rem;color:var(--txt);line-height:1.7;}
/* DCT & RLE VISUALIZATIONS */
.dct-wrapper{display:flex;flex-direction:column;align-items:center;background:var(--bg-1);padding:1.5rem;border-radius:8px;border:1px solid var(--border);margin:1rem 0;}
.dct-grid{display:grid;grid-template-columns:repeat(8,1fr);gap:2px;width:100%;max-width:320px;background:var(--border);padding:2px;border-radius:4px;}
.dct-cell{aspect-ratio:1;display:flex;align-items:center;justify-content:center;font-size:0.65rem;font-family:'IBM Plex Mono',monospace;border-radius:2px;font-weight:600;}
.dct-title{font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:var(--cyan);margin-bottom:1rem;text-transform:uppercase;letter-spacing:0.1em;}
.rle-wrapper{display:flex;flex-wrap:wrap;gap:8px;background:var(--bg-1);padding:1.25rem;border-radius:8px;border:1px solid var(--border);margin:1rem 0;}
.rle-pill{display:flex;border-radius:6px;overflow:hidden;border:1px solid var(--border);}
.rle-count{background:linear-gradient(135deg,var(--cyan),var(--violet));color:#fff;font-weight:800;padding:4px 10px;font-size:0.8rem;font-family:'IBM Plex Mono',monospace;display:flex;align-items:center;justify-content:center;}
.rle-val{background:var(--bg-3);color:var(--txt);padding:4px 12px;font-size:0.8rem;font-family:'IBM Plex Mono',monospace;display:flex;align-items:center;justify-content:center;}
/* CODE BLOCKS */
.stCode,[data-testid="stCode"]{background:#010508 !important;border:1px solid rgba(6,182,212,0.18) !important;border-radius:var(--radius-md) !important;}
code{color:var(--cyan) !important;background:var(--bg-2) !important;border:1px solid var(--border) !important;border-radius:4px;padding:1px 5px;}
/* NO-FILE STATE */
.no-file-state{text-align:center;padding:3rem 2rem;background:var(--bg-2);border:1.5px dashed var(--border);border-radius:var(--radius-lg);margin-top:1.5rem;}
.no-file-state .nfs-icon{font-size:2.5rem;margin-bottom:0.75rem;}
.no-file-state .nfs-title{font-size:1rem;font-weight:600;color:var(--txt-dim);margin-bottom:0.4rem;}
.no-file-state .nfs-sub{font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:var(--muted);}

/* ═══ CANAL LAB: MATRIX FEC & STEPS ═══ */
.fec-step-container {background:var(--bg-1); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;}
.fec-step-title {font-family:'IBM Plex Mono',monospace; font-size: 0.85rem; color: var(--cyan); margin-bottom: 10px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; border-bottom: 1px solid var(--border); padding-bottom: 5px;}
.fec-step-desc {font-size: 0.8rem; color: var(--txt-dim); margin-bottom: 15px;}
.matrix-grid {display: inline-flex; flex-direction: column; gap: 2px; padding: 10px; background: var(--bg-2); border-radius: 6px; border: 1px solid var(--border-strong);}
.matrix-row {display: flex; gap: 2px;}
.matrix-cell{display:flex;align-items:center;justify-content:center;width:28px;height:28px;border:1px solid rgba(255,255,255,0.05);font-family:'IBM Plex Mono',monospace;font-size:13px;border-radius:4px; transition: all 0.3s;}
.data-bit{background-color:#1e3a8a;color:#cdd9e5;}
.parity-bit-h{background-color:#065f46;color:#a7f3d0;font-weight:bold; box-shadow: inset 0 0 5px rgba(0,0,0,0.5);}
.parity-bit-v{background-color:#0d9488;color:#ccfbf1;font-weight:bold; box-shadow: inset 0 0 5px rgba(0,0,0,0.5);}
.parity-bit-m{background-color:#9333ea;color:#f3e8ff;font-weight:bold; box-shadow: inset 0 0 8px rgba(0,0,0,0.8);}
.error-bit{background-color:#991b1b;color:white;font-weight:bold;animation:blinker 1s linear infinite; border: 2px solid #ef4444;}
.fixed-bit{background-color:#16a34a;color:white;font-weight:bold; border: 2px solid #4ade80;}
.empty-cell{background-color:transparent; border:none;}
@keyframes blinker{50%{opacity:0.3;}}

/* DOWNLOAD BUTTON */
.dl-btn{display:block;background:linear-gradient(135deg,var(--violet) 0%,#7b2cbf 100%);color:white !important;padding:12px 20px;border-radius:8px;font-weight:bold;text-decoration:none;text-align:center;margin-top:15px;transition:0.3s;font-family:'IBM Plex Mono',monospace;font-size:0.85rem; box-shadow: 0 4px 15px rgba(139,92,246,0.2);}
.dl-btn:hover{box-shadow:0 6px 20px rgba(139,92,246,0.4); transform: translateY(-2px);}
</style>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  CODE MAP (Fuente Lab reference snippets)
# ═════════════════════════════════════════════════════════════════════════════
_CODE_MAP: Dict[str, str] = {
    "Huffman": """\
# Fórmula Entropía: H(X) = -Σ p(x) * log2(p(x))
def construir_arbol(freq: dict):
    heap = [Nodo(s, f) for s, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        izq, der = heapq.heappop(heap), heapq.heappop(heap)
        padre = Nodo(None, izq.frec + der.frec)
        padre.izq, padre.der = izq, der
        heapq.heappush(heap, padre)
    return heap[0]
""",
    "LZW": """\
# Compresión por Diccionario LZW
def lzw_comprimir(texto: str):
    dic = {chr(i): i for i in range(256)}
    sig = 256
    w, salida = "", []
    for c in texto:
        wc = w + c
        if wc in dic: w = wc
        else:
            salida.append(dic[w])
            dic[wc] = sig; sig += 1
            w = c
    if w: salida.append(dic[w])
    return salida
""",
    "RLE": """\
# Run-Length Encoding (RLE)
def rle_comprimir(datos: bytes) -> bytes:
    out, i = bytearray(), 0
    while i < len(datos):
        byte = datos[i]; run = 1
        while i + run < len(datos) and datos[i+run] == byte and run < 255:
            run += 1
        out.extend([run, byte]); i += run
    return bytes(out)
""",
    "DCT — JPEG-like": """\
# Transformada Discreta del Coseno (DCT-II 2D)
def dct2(bloque: np.ndarray) -> np.ndarray:
    N = 8; b = bloque.astype(float) - 128
    F = np.zeros((N, N)); xs = np.arange(N)
    for u in range(N):
        cu = 1/math.sqrt(2) if u == 0 else 1.0
        cos_u = np.cos((2*xs+1)*u*math.pi/(2*N))
        for v in range(N):
            cv = 1/math.sqrt(2) if v == 0 else 1.0
            cos_v = np.cos((2*xs+1)*v*math.pi/(2*N))
            F[u,v] = (2/N)*cu*cv*np.sum(b*cos_u[:,None]*cos_v[None,:])
    return F
""",
    "H.264 / HEVC (Simulado)": """\
# Estimación de Movimiento (Motion Estimation)
def motion_estimation(frame_ref, frame_curr, block_size=16):
    motion_vectors = []
    for i in range(0, frame_curr.shape[0], block_size):
        for j in range(0, frame_curr.shape[1], block_size):
            curr_block = frame_curr[i:i+block_size, j:j+block_size]
            best_sad, best_mv = float('inf'), (0, 0)
            for dy in range(-16, 17):
                for dx in range(-16, 17):
                    sad = np.sum(np.abs(curr_block - frame_ref[i+dy:..., j+dx:...]))
                    if sad < best_sad: best_sad, best_mv = sad, (dx, dy)
            motion_vectors.append(best_mv)
    return motion_vectors
""",
    "μ-Law G.711": """\
# Companding Logarítmico μ-Law
def encode_mulaw(sample: int) -> int:
    MU, BIAS = 255, 132
    sample = max(-32768, min(32767, sample))
    sign = 0x00 if sample >= 0 else 0x80
    mag = min(abs(sample), 32635) + BIAS
    exp = max(0, min(int(math.log2(mag)) - 7, 7))
    mant = (mag >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mant)) & 0xFF
""",
    "ADPCM": """\
# IMA ADPCM — Adaptive Differential PCM
# Δ = (x_n - x_pred) / step_size
""",
}


# ═════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES & ANALYZERS (Fuente Lab)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class EstadisticasInfo:
    frecuencias: Dict[str, int]
    probabilidades: Dict[str, float]
    entropia: float
    longitud_promedio: float
    eficiencia: float
    redundancia: float
    total_simbolos: int
    simbolos_unicos: int
    entropia_maxima: float


@dataclass
class ResultadoCompresion:
    nombre_algoritmo: str
    datos_originales: bytes
    datos_comprimidos: bytes
    datos_descomprimidos: bytes
    tamaño_original: int
    tamaño_comprimido: int
    tasa_compresion: float
    ratio_reduccion: float
    tiempo_ms: float
    pasos: List[Dict[str, Any]] = field(default_factory=list)
    tabla_codigos: Optional[Dict[str, str]] = None
    tabla_lzw: Optional[List[Dict]] = None
    grafo_dot: Optional[str] = None
    es_stub: bool = True

class AnalizadorInformacion(ABC):
    def __init__(self, datos: bytes, nombre: str = "datos") -> None:
        if not datos: raise ValueError("Los datos no pueden estar vacíos.")
        self._datos = datos
        self._nombre = nombre

    @abstractmethod
    def _extraer_simbolos(self) -> List[Any]: ...

    def calcular_frecuencias(self) -> Dict[str, int]:
        return dict(Counter(self._extraer_simbolos()))

    def calcular_probabilidades(self) -> Dict[str, float]:
        freq = self.calcular_frecuencias()
        total = sum(freq.values())
        return {k: v / total for k, v in freq.items()}

    def calcular_entropia(self) -> float:
        return -sum(p * math.log2(p) for p in self.calcular_probabilidades().values() if p > 0)

    def calcular_longitud_promedio(self, longitudes: Optional[Dict[str, int]] = None) -> float:
        probs = self.calcular_probabilidades()
        if longitudes:
            return sum(probs[s] * longitudes[s] for s in probs if s in longitudes)
        return sum(p * math.ceil(-math.log2(p)) for p in probs.values() if p > 0)

    def calcular_eficiencia(self, L_bar: Optional[float] = None) -> float:
        H = self.calcular_entropia()
        L = L_bar if L_bar is not None else self.calcular_longitud_promedio()
        return H / L if L > 0 else 0.0

    def calcular_todo(self) -> EstadisticasInfo:
        freq = self.calcular_frecuencias()
        probs = self.calcular_probabilidades()
        H = self.calcular_entropia()
        L = self.calcular_longitud_promedio()
        eta = self.calcular_eficiencia(L)
        n = len(freq)
        return EstadisticasInfo(freq, probs, H, L, eta, 1.0 - eta, sum(freq.values()), n, math.log2(n) if n > 1 else 0.0)

class AnalizadorTexto(AnalizadorInformacion):
    def _extraer_simbolos(self) -> List[str]:
        try: return list(self._datos.decode("utf-8", errors="replace"))
        except Exception: return list(self._datos.decode("latin-1", errors="replace"))

class AnalizadorImagen(AnalizadorInformacion):
    def _extraer_simbolos(self) -> List[int]: return list(self._datos)

class AnalizadorAudio(AnalizadorInformacion):
    def _extraer_simbolos(self) -> List[int]: return list(self._datos)

class AnalizadorVideo(AnalizadorInformacion):
    CAP = 100_000
    def _extraer_simbolos(self) -> List[int]: return list(self._datos[:self.CAP])


# ═════════════════════════════════════════════════════════════════════════════
#  CANAL LAB: UTILITY FUNCTIONS & Modulator
# ═════════════════════════════════════════════════════════════════════════════
def bytes_to_bits(data: bytes) -> str:
    return ''.join(format(b, '08b') for b in data)

def bits_to_bytes(bits: str) -> bytes:
    return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8) if i + 8 <= len(bits))

def create_download_link(payload_dict: dict, filename: str) -> str:
    json_str = json.dumps(payload_dict, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    return (f'<a href="data:application/octet-stream;base64,{b64}" '
            f'download="{filename}" class="dl-btn">⬇ Descargar Payload .BIN ({filename})</a>')

def inject_bit_errors(bits: str, ber: float) -> str:
    if ber <= 0: return bits
    bit_list = list(bits)
    for _ in range(int(len(bit_list) * (ber / 1.5))):
        idx = random.randint(0, len(bit_list) - 1)
        bit_list[idx] = '1' if bit_list[idx] == '0' else '0'
    return "".join(bit_list)

def recommend_modulation(_: int) -> str:
    return ("💡 **Recomendación**: La Paridad 2D es robusta para detectar y corregir errores simples. "
            "Se recomienda **QPSK** para un balance velocidad/robustez, o **BPSK** en canales muy ruidosos.")

class Modulator:
    def __init__(self, scheme: str) -> None:
        self.scheme = scheme

    def modulate(self, bits: str) -> np.ndarray:
        if not bits: return np.array([])
        if self.scheme == 'BPSK':
            return np.array([1 if b == '1' else -1 for b in bits])
        elif self.scheme == 'QPSK':
            mapping = {'00': (-1, -1), '01': (-1, 1), '10': (1, -1), '11': (1, 1)}
            return np.array([mapping[bits[i:i+2].ljust(2, '0')] for i in range(0, len(bits), 2)])
        else:  # QAM16
            levels = [-3, -1, 1, 3]
            return np.array([
                (levels[int(bits[i:i+4].ljust(4, '0')[:2], 2)],
                 levels[int(bits[i:i+4].ljust(4, '0')[2:], 2)])
                for i in range(0, len(bits), 4)
            ])

    def channel_awgn(self, signal: np.ndarray, ber: float) -> np.ndarray:
        return signal + np.random.normal(0, ber * 2.0, signal.shape)

    def render_constellation(self, noisy_signal: np.ndarray) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor('#0a0a0f')
        ax.set_facecolor('#111827')
        for spine in ax.spines.values(): spine.set_edgecolor('#374151')
        ax.tick_params(colors='#8b9cb5')
        if self.scheme == 'BPSK':
            ax.scatter(noisy_signal, np.zeros(len(noisy_signal)), c='#06b6d4', s=10, alpha=0.5)
        else:
            ax.scatter(noisy_signal[:, 0], noisy_signal[:, 1], c='#8b5cf6', s=10, alpha=0.5)
        ax.set_title(f"Constelación {self.scheme}", color='#cdd9e5', fontsize=10)
        ax.axhline(0, color='#374151', lw=0.5)
        ax.axvline(0, color='#374151', lw=0.5)
        fig.tight_layout()
        return fig

# ═════════════════════════════════════════════════════════════════════════════
#  ENHANCED MATRIX FEC 2D (Paso a Paso)
# ═════════════════════════════════════════════════════════════════════════════
class MatrixFEC:
    def __init__(self, cols: int = 4) -> None:
        self.cols = cols

    def calculate_parity(self, bit_string: str) -> str:
        # Paridad Par: Retorna 1 si hay un número impar de 1s (para hacer que la suma total de 1s sea par)
        return '1' if bit_string.count('1') % 2 != 0 else '0'

    def encode(self, bits: str):
        if not bits: return "", [], "", 0
        padding_needed = (self.cols - (len(bits) % self.cols)) % self.cols
        padded_bits = bits + ('0' * padding_needed)
        rows = len(padded_bits) // self.cols
        
        matrix = []
        encoded_stream = ""
        col_parities = ['0'] * self.cols

        # Construimos el HTML Paso a Paso para la Transmisión
        html_steps = "<div class='fec-visualizer'>"
        
        # PASO 1: Bloques de Datos
        html_steps += "<div class='fec-step-container'><div class='fec-step-title'>Paso 1: División en Trama de Datos</div>"
        html_steps += "<div class='fec-step-desc'>El flujo de bits se divide en filas del ancho configurado (" + str(self.cols) + "). Se añade padding si es necesario.</div>"
        html_steps += "<div class='matrix-grid'>"
        for r in range(rows):
            row_data = padded_bits[r * self.cols:(r + 1) * self.cols]
            html_steps += "<div class='matrix-row'>"
            for bit in row_data:
                html_steps += f"<div class='matrix-cell data-bit'>{bit}</div>"
            html_steps += "</div>"
        html_steps += "</div></div>"

        # PASO 2: Paridad Horizontal
        html_steps += "<div class='fec-step-container'><div class='fec-step-title'>Paso 2: Paridad Horizontal (Filas)</div>"
        html_steps += "<div class='fec-step-desc'>Se calcula el bit de paridad par para cada fila (se añade a la derecha).</div>"
        html_steps += "<div class='matrix-grid'>"
        
        for r in range(rows):
            row_data = padded_bits[r * self.cols:(r + 1) * self.cols]
            row_parity = self.calculate_parity(row_data)
            matrix.append(list(row_data) + [row_parity])
            
            html_steps += "<div class='matrix-row'>"
            for bit in row_data:
                html_steps += f"<div class='matrix-cell data-bit'>{bit}</div>"
            html_steps += f"<div class='matrix-cell parity-bit-h'>{row_parity}</div>"
            html_steps += "</div>"
            
            # Update column parities logic
            for c in range(self.cols):
                if row_data[c] == '1': col_parities[c] = '0' if col_parities[c] == '1' else '1'
            
            encoded_stream += row_data + row_parity
            
        html_steps += "</div></div>"

        # PASO 3: Paridad Vertical & Master
        html_steps += "<div class='fec-step-container'><div class='fec-step-title'>Paso 3: Paridad Vertical y Paridad Maestra</div>"
        html_steps += "<div class='fec-step-desc'>Se calcula la paridad por columnas (abajo) y finalmente el bit que cruza ambas paridades (esquina inferior derecha). Este bloque completo se serializa para transmitir.</div>"
        html_steps += "<div class='matrix-grid'>"
        
        for r in range(rows):
            html_steps += "<div class='matrix-row'>"
            for bit in matrix[r][:-1]: html_steps += f"<div class='matrix-cell data-bit'>{bit}</div>"
            html_steps += f"<div class='matrix-cell parity-bit-h'>{matrix[r][-1]}</div>"
            html_steps += "</div>"

        # Last row for column parities
        master_parity = self.calculate_parity("".join(col_parities))
        html_steps += "<div class='matrix-row' style='margin-top:2px;'>"
        for cp in col_parities:
            html_steps += f"<div class='matrix-cell parity-bit-v'>{cp}</div>"
        html_steps += f"<div class='matrix-cell parity-bit-m'>{master_parity}</div>"
        html_steps += "</div></div></div></div>"

        encoded_stream += "".join(col_parities) + master_parity
        return encoded_stream, matrix, html_steps, padding_needed

    def decode_and_correct(self, rx_stream: str, padding_needed: int):
        if not rx_stream: return "", "", [], "Sin datos."
        row_len = self.cols + 1
        num_data_rows = (len(rx_stream) // row_len) - 1
        rx_matrix = []
        idx = 0
        
        for _ in range(num_data_rows + 1):
            rx_matrix.append(list(rx_stream[idx:idx + row_len]))
            idx += row_len

        error_row, error_col = -1, -1
        syndrome_logs = []

        # Paso a Paso visualización Recepción
        html_rx = "<div class='fec-visualizer'>"
        
        # PASO 1: Matriz Recibida
        html_rx += "<div class='fec-step-container'><div class='fec-step-title'>1. Matriz con Ruido (AWGN)</div>"
        html_rx += "<div class='fec-step-desc'>Esta es la trama recibida, reorganizada en su estructura 2D.</div>"
        html_rx += "<div class='matrix-grid'>"
        for r in range(len(rx_matrix)):
            html_rx += "<div class='matrix-row'>"
            for c in range(len(rx_matrix[r])):
                is_p_row = (r == num_data_rows)
                is_p_col = (c == self.cols)
                css = "data-bit"
                if is_p_row and is_p_col: css = "parity-bit-m"
                elif is_p_row: css = "parity-bit-v"
                elif is_p_col: css = "parity-bit-h"
                html_rx += f"<div class='matrix-cell {css}'>{rx_matrix[r][c]}</div>"
            html_rx += "</div>"
        html_rx += "</div></div>"

        # Cálculo Síndromes
        for r in range(num_data_rows):
            row_data = "".join(rx_matrix[r][:-1])
            exp_p = self.calculate_parity(row_data)
            rx_p = rx_matrix[r][-1]
            if exp_p != rx_p:
                error_row = r
                syndrome_logs.append(f"❌ **Síndrome Fila {r}:** Esperado `{exp_p}`, recibido `{rx_p}`.")

        for c in range(self.cols):
            col_data = "".join([rx_matrix[r][c] for r in range(num_data_rows)])
            exp_p = self.calculate_parity(col_data)
            rx_p = rx_matrix[-1][c]
            if exp_p != rx_p:
                error_col = c
                syndrome_logs.append(f"❌ **Síndrome Col {c}:** Esperado `{exp_p}`, recibido `{rx_p}`.")

        correction_log = "✅ Síndrome 0: Trama limpia. No se detectaron errores."
        
        if error_row != -1 and error_col != -1:
            bad_bit = rx_matrix[error_row][error_col]
            corrected_bit = '0' if bad_bit == '1' else '1'
            correction_log = f"🎯 **Cruce de Síndromes:** Error detectado en Coordenada ({error_row},{error_col}). Bit corregido: `{bad_bit}` ➔ `{corrected_bit}`."
            
            html_rx += "<div class='fec-step-container'><div class='fec-step-title'>2. Cruce de Coordenadas y Corrección</div>"
            html_rx += f"<div class='fec-step-desc'>{correction_log}</div>"
            html_rx += "<div class='matrix-grid'>"
            
            for r in range(len(rx_matrix)):
                html_rx += "<div class='matrix-row'>"
                for c in range(len(rx_matrix[r])):
                    is_err_target = (r == error_row and c == error_col)
                    is_crosshair = (r == error_row or c == error_col)
                    
                    css = "data-bit"
                    if is_err_target: css = "error-bit"
                    elif r == num_data_rows and c == self.cols: css = "parity-bit-m"
                    elif r == num_data_rows: css = "parity-bit-v"
                    elif c == self.cols: css = "parity-bit-h"
                    
                    # Highlight crosshair slightly
                    if is_crosshair and not is_err_target:
                        html_rx += f"<div class='matrix-cell {css}' style='opacity:0.6'>{rx_matrix[r][c]}</div>"
                    else:
                        html_rx += f"<div class='matrix-cell {css}'>{rx_matrix[r][c]}</div>"
                html_rx += "</div>"
            html_rx += "</div></div>"
            
            # Apply correction
            rx_matrix[error_row][error_col] = corrected_bit
            
            # Show fixed matrix
            html_rx += "<div class='fec-step-container'><div class='fec-step-title'>3. Matriz Corregida y Lista para Decode</div>"
            html_rx += "<div class='matrix-grid'>"
            for r in range(len(rx_matrix)):
                html_rx += "<div class='matrix-row'>"
                for c in range(len(rx_matrix[r])):
                    css = "data-bit"
                    if r == error_row and c == error_col: css = "fixed-bit"
                    elif r == num_data_rows and c == self.cols: css = "parity-bit-m"
                    elif r == num_data_rows: css = "parity-bit-v"
                    elif c == self.cols: css = "parity-bit-h"
                    html_rx += f"<div class='matrix-cell {css}'>{rx_matrix[r][c]}</div>"
                html_rx += "</div>"
            html_rx += "</div></div>"
            
        elif len(syndrome_logs) > 0:
            # Errors exist but couldn't fix perfectly (e.g. multi-bit in same row/col)
             html_rx += "<div class='fec-step-container'><div class='fec-step-title'>⚠️ Advertencia: Errores Multi-Bit</div>"
             html_rx += "<div class='fec-step-desc'>La cantidad de errores supera la capacidad de corrección de la Paridad 2D.</div></div>"
        else:
             html_rx += "<div class='fec-step-container'><div class='fec-step-title'>✅ Trama Limpia</div>"
             html_rx += "<div class='fec-step-desc'>Síndrome 0. La trama superó el canal sin mutaciones de bits.</div></div>"

        html_rx += "</div>"

        clean_bits = "".join("".join(rx_matrix[r][:-1]) for r in range(num_data_rows))
        if padding_needed > 0:
            clean_bits = clean_bits[:-padding_needed]
            
        return clean_bits, html_rx, syndrome_logs, correction_log

# ═════════════════════════════════════════════════════════════════════════════
#  OTHER LAB UTILITIES (Huffman, DCT, MuLaw, etc)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(order=True)
class HuffNode:
    freq: int
    symbol: Any = field(compare=False)
    left: Any = field(compare=False, default=None)
    right: Any = field(compare=False, default=None)

class HuffmanCoderCanal:
    def encode(self, text: str):
        freq = Counter(text)
        heap = [HuffNode(v, k) for k, v in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            l, r = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, HuffNode(l.freq + r.freq, None, l, r))
        codes: Dict[str, str] = {}
        def gen_codes(n: HuffNode, current: str = '') -> None:
            if n.symbol is not None: codes[n.symbol] = current or '0'
            if n.left: gen_codes(n.left, current + '0')
            if n.right: gen_codes(n.right, current + '1')
        if heap: gen_codes(heap[0])
        encoded = ''.join(codes[ch] for ch in text)
        inverse = {v: k for k, v in codes.items()}
        return encoded, inverse, heap[0] if heap else None, freq

    def decode_visual_log(self, encoded: str, inverse: Dict[str, str], max_logs: int = 10):
        curr, out, logs = '', '', []
        for b in encoded:
            curr += b
            if curr in inverse:
                if len(logs) < max_logs:
                    logs.append({"Buffer Rx": curr, "Match": "✅", "Símbolo": inverse[curr]})
                out += inverse[curr]
                curr = ''
        return out, logs

class MuLawCodec:
    MU = 255
    @staticmethod
    def encode(samples: np.ndarray) -> np.ndarray:
        s_norm = samples.astype(np.float32) / 32768.0
        s_comp = np.sign(s_norm) * (np.log1p(MuLawCodec.MU * np.abs(s_norm)) / np.log1p(MuLawCodec.MU))
        return np.int8(s_comp * 127)

    @staticmethod
    def decode(encoded: np.ndarray) -> np.ndarray:
        s_norm = encoded.astype(np.float32) / 127.0
        s_exp = np.sign(s_norm) * (1 / MuLawCodec.MU) * (np.power(1 + MuLawCodec.MU, np.abs(s_norm)) - 1)
        return np.int16(s_exp * 32767.0)

def process_full_image_dct(img_arr: np.ndarray):
    h, w = img_arr.shape
    h_pad, w_pad = (8 - h % 8) % 8, (8 - w % 8) % 8
    padded = np.pad(img_arr, ((0, h_pad), (0, w_pad)), mode='constant')
    dct_blocks = np.zeros_like(padded, dtype=float)
    for i in range(0, padded.shape[0], 8):
        for j in range(0, padded.shape[1], 8):
            block = padded[i:i+8, j:j+8]
            dct_blocks[i:i+8, j:j+8] = np.round(scipy_dct(scipy_dct(block.T, norm='ortho').T, norm='ortho') / 10)
    return dct_blocks, padded.shape

def reconstruct_full_image_idct(dct_blocks: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    idct_blocks = np.zeros_like(dct_blocks, dtype=float)
    for i in range(0, dct_blocks.shape[0], 8):
        for j in range(0, dct_blocks.shape[1], 8):
            block = dct_blocks[i:i+8, j:j+8]
            idct_blocks[i:i+8, j:j+8] = scipy_idct(scipy_idct((block * 10).T, norm='ortho').T, norm='ortho')
    return np.clip(idct_blocks[:orig_h, :orig_w], 0, 255).astype(np.uint8)

# ═════════════════════════════════════════════════════════════════════════════
#  CANAL LAB: TAB RENDERERS
# ═════════════════════════════════════════════════════════════════════════════
def render_canal_texto() -> None:
    col_tx, col_rx = st.columns(2)

    with col_tx:
        st.markdown("<div class='section-label'><span class='sl-icon'>📤</span><span class='sl-text'>MÓDULO TRANSMISOR</span></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: mod_txt = st.selectbox("Modulación", ["QPSK", "BPSK", "QAM16"], key="c_tx_mod")
        with c2: fec_cols = st.selectbox("Ancho FEC", [4, 8, 16], key="c_tx_fec")
        with c3: ber_txt = st.slider("BER AWGN", 0.0, 1.0, 0.0, key="c_tx_ber")
        st.info(recommend_modulation(fec_cols))
        text_input = st.text_area("Payload:", "HOLA MUNDO", key="c_txt_payload")

        if st.button("▶ Ejecutar Pipeline Tx", type="primary", key="c_btn_tx_txt"):
            tx_bits, inverse, _, _ = HuffmanCoderCanal().encode(text_input)
            
            with st.expander("🛡️ Análisis FEC 2D (Paso a Paso)", expanded=True):
                fec = MatrixFEC(cols=fec_cols)
                fec_bits, _, tx_html, padding = fec.encode(tx_bits)
                st.markdown(tx_html, unsafe_allow_html=True)
                
            with st.expander("📡 Constelación + AWGN", expanded=True):
                mod = Modulator(mod_txt)
                signal = mod.modulate(fec_bits)
                if len(signal) > 0:
                    st.pyplot(mod.render_constellation(mod.channel_awgn(signal, ber_txt)))
                    
            meta = {"inverse": inverse, "alg": "Huffman", "cols": fec_cols, "padding": padding}
            payload = {"modulo": "texto", "metadata": meta, "original_fec_bits": fec_bits, "rx_bits": inject_bit_errors(fec_bits, ber_txt)}
            # Mantener la funcionalidad de descarga para el payload codificado
            st.markdown(create_download_link(payload, "texto_canal.bin"), unsafe_allow_html=True)

    with col_rx:
        st.markdown("<div class='section-label'><span class='sl-icon'>📥</span><span class='sl-text'>MÓDULO RECEPTOR</span></div>", unsafe_allow_html=True)
        rx_file = st.file_uploader("Sube .bin", type=["bin", "json"], key="c_rx_txt")
        if rx_file:
            data = json.load(rx_file)
            with st.expander("🛠️ Reconstrucción y Síndromes FEC", expanded=True):
                fec = MatrixFEC(cols=data["metadata"]["cols"])
                source_bits, rx_html, syndromes, correction = fec.decode_and_correct(data["rx_bits"], data["metadata"]["padding"])
                
                # Render the step-by-step decoding visualizer
                st.markdown(rx_html, unsafe_allow_html=True)
                
                # Display raw text logs below visualizer
                st.markdown("**Logs de Diagnóstico del Canal:**")
                if syndromes:
                    for s in syndromes: st.write(s)
                else:
                    st.write("Ningún síndrome alterado. Señal perfecta.")
                    
            with st.expander("🧩 Decodificación Huffman", expanded=True):
                try:
                    res_txt, logs = HuffmanCoderCanal().decode_visual_log(source_bits, data["metadata"]["inverse"])
                    st.markdown(f"> **OUTPUT FINAL:** `{res_txt}`")
                    if logs: st.dataframe(pd.DataFrame(logs), use_container_width=True)
                except Exception as e:
                    st.error(f"Error de decodificación: {e}")

# The rest of the lab renderer functions (render_canal_imagen, render_canal_audio, etc.) 
# will naturally inherit the detailed FEC visualization since they call MatrixFEC.decode_and_correct as well!

def main() -> None:
    inject_css()
    # Simplified main for brevity, in your complete file you would just hook up the tabs as before.
    st.markdown("""<div class="app-header-wrap"><div class="app-header"><div class="header-icon">📡</div><div><p class="app-title">Telecom DSP Lab · Fuente &amp; Canal</p><p class="app-subtitle">Shannon · Huffman · LZW · RLE · DCT · μ-Law · ADPCM · FEC Matricial 2D · BPSK / QPSK / QAM16</p></div></div></div>""", unsafe_allow_html=True)
    lab_tabs = st.tabs(["📡 Lab Canal · FEC & Modulación"])
    
    with lab_tabs[0]:
        st.markdown("<div class='info-box violet'><div class='ib-icon'>📡</div><div class='ib-content'>Pipeline completo <strong>Tx / Rx</strong>: Huffman → <strong>FEC Matricial 2D (Paso a Paso)</strong> → Modulación <strong>BPSK / QPSK / QAM16</strong> → Canal <strong>AWGN</strong> → Reconstrucción.</div></div>", unsafe_allow_html=True)
        render_canal_texto()

if __name__ == "__main__":
    main()
