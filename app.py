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
/* ═══ CANAL LAB: MATRIX FEC ═══ */
.matrix-cell{display:inline-block;width:26px;height:26px;line-height:26px;text-align:center;border:1px solid #444;margin:1px;font-family:'IBM Plex Mono',monospace;font-size:13px;border-radius:3px;}
.data-bit{background-color:#1e3a8a;color:#cdd9e5;}
.parity-bit{background-color:#065f46;color:#a7f3d0;font-weight:bold;}
.error-bit{background-color:#991b1b;color:white;font-weight:bold;animation:blinker 1s linear infinite;}
@keyframes blinker{50%{opacity:0;}}
/* DOWNLOAD BUTTON */
.dl-btn{display:block;background:linear-gradient(135deg,var(--violet) 0%,#7b2cbf 100%);color:white !important;padding:10px 20px;border-radius:8px;font-weight:bold;text-decoration:none;text-align:center;margin-top:10px;transition:0.3s;font-family:'IBM Plex Mono',monospace;font-size:0.8rem;}
.dl-btn:hover{box-shadow:0 0 15px var(--violet);}
</style>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  CODE MAP (Fuente Lab reference snippets)
# ═════════════════════════════════════════════════════════════════════════════
_CODE_MAP: Dict[str, str] = {
    "Huffman": """\
# Fórmula Entropía: H(X) = -Σ p(x) * log2(p(x))
# Longitud L̄ = Σ p(x) * len(codigo)

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
# F(u,v) = (2/N)*Cu*Cv * ΣΣ f(x,y)*cos(...)*cos(...)

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
# SAD(mv) = ΣΣ |frame_curr(x,y) - frame_ref(x+mvx, y+mvy)|

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
# y = sgn(x) * ln(1 + μ|x|) / ln(1 + μ)

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
#  DATA CLASSES (Fuente Lab)
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


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYZERS (Fuente Lab)
# ═════════════════════════════════════════════════════════════════════════════
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
        return EstadisticasInfo(freq, probs, H, L, eta, 1.0 - eta,
                                sum(freq.values()), n, math.log2(n) if n > 1 else 0.0)


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
#  CANAL LAB: UTILITY FUNCTIONS
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


def create_bytes_download_link(data: bytes, filename: str, label: str = None) -> str:
    b64 = base64.b64encode(data).decode()
    lbl = label or f"⬇ Descargar {filename}"
    ext = filename.rsplit('.', 1)[-1].upper()
    mime = {
        "BIN": "application/octet-stream",
        "TXT": "text/plain",
        "WAV": "audio/wav",
        "JSON": "application/json",
    }.get(ext, "application/octet-stream")
    return (f'<a href="data:{mime};base64,{b64}" '
            f'download="{filename}" class="dl-btn">{lbl} ({filename})</a>')


def create_text_download_link(text: str, filename: str, label: str = None) -> str:
    return create_bytes_download_link(text.encode("utf-8"), filename, label)


# ─── FEC VISUAL STEP-BY-STEP ────────────────────────────────────────────────
def render_fec_visual_steps(bits: str, cols: int = 4, ber: float = 0.05) -> dict:
    """
    Genera HTML detallado para los 5 pasos de FEC Matricial 2D.
    Retorna dict con html de cada paso y métricas de canal.
    """
    # ── Preparar datos ──
    padding_needed = (cols - (len(bits) % cols)) % cols
    padded = bits + '0' * padding_needed
    rows = len(padded) // cols

    def calc_par(s: str) -> str:
        return '1' if s.count('1') % 2 != 0 else '0'

    # Construir matriz de datos + paridades de fila
    matrix: List[List[str]] = []
    for r in range(rows):
        row = list(padded[r * cols:(r + 1) * cols])
        row_p = calc_par("".join(row))
        matrix.append(row + [row_p])

    # Paridades de columna
    col_pars = []
    for c in range(cols):
        col_data = "".join(matrix[r][c] for r in range(rows))
        col_pars.append(calc_par(col_data))
    master = calc_par("".join(col_pars))
    col_pars.append(master)
    matrix.append(col_pars)

    num_data_rows = rows

    cell = lambda val, cls, extra="": (
        f'<span class="matrix-cell {cls}" style="{extra}">{val}</span>'
    )

    # ── PASO 1: Matriz Original de datos ──
    html_p1 = "<div style='margin:8px 0'>"
    for r in range(num_data_rows):
        for c in range(cols):
            html_p1 += cell(matrix[r][c], "data-bit")
        html_p1 += "<br>"
    html_p1 += "</div>"

    # ── PASO 2: Matriz con Paridades ──
    html_p2 = "<div style='margin:8px 0'>"
    for r in range(num_data_rows):
        for c in range(cols):
            html_p2 += cell(matrix[r][c], "data-bit")
        html_p2 += cell(matrix[r][cols], "parity-bit") + "<br>"
    # fila col-paridades
    for c in range(cols):
        html_p2 += cell(col_pars[c], "parity-bit")
    html_p2 += cell(master, "parity-bit", "background:#9333ea;") + "<br>"
    html_p2 += "</div>"

    # ── PASO 3: Inyectar error ──
    # Elegimos una posición de datos al azar (reproducible con la semilla)
    rng = random.Random(42)
    err_r = rng.randint(0, num_data_rows - 1)
    err_c = rng.randint(0, cols - 1)
    rx_matrix = [row[:] for row in matrix]
    original_bit = rx_matrix[err_r][err_c]
    rx_matrix[err_r][err_c] = '0' if original_bit == '1' else '1'

    html_p3 = "<div style='margin:8px 0'>"
    for r in range(num_data_rows):
        for c in range(cols):
            if r == err_r and c == err_c:
                html_p3 += cell(rx_matrix[r][c], "error-bit")
            else:
                html_p3 += cell(rx_matrix[r][c], "data-bit")
        html_p3 += cell(rx_matrix[r][cols], "parity-bit") + "<br>"
    for c in range(cols):
        html_p3 += cell(col_pars[c], "parity-bit")
    html_p3 += cell(master, "parity-bit", "background:#9333ea;") + "<br>"
    html_p3 += "</div>"

    # ── PASO 4: Síndrome ──
    syn_row, syn_col = -1, -1
    syndrome_details = []
    for r in range(num_data_rows):
        row_data = "".join(rx_matrix[r][:cols])
        exp_p = calc_par(row_data)
        rx_p = rx_matrix[r][cols]
        xor_val = int(exp_p) ^ int(rx_p)
        color = "#ef4444" if xor_val else "#10b981"
        syndrome_details.append({
            "fila": r, "esperado": exp_p, "recibido": rx_p,
            "xor": xor_val, "color": color
        })
        if exp_p != rx_p:
            syn_row = r

    col_syn_details = []
    for c in range(cols):
        col_data = "".join(rx_matrix[r][c] for r in range(num_data_rows))
        exp_p = calc_par(col_data)
        rx_p = col_pars[c]
        xor_val = int(exp_p) ^ int(rx_p)
        color = "#ef4444" if xor_val else "#10b981"
        col_syn_details.append({
            "col": c, "esperado": exp_p, "recibido": rx_p,
            "xor": xor_val, "color": color
        })
        if exp_p != rx_p:
            syn_col = c

    M = "#1a2540"
    html_p4 = (
        "<div style='font-family:\"IBM Plex Mono\",monospace;font-size:0.75rem;"
        "background:var(--bg-1);padding:1rem;border-radius:8px;border:1px solid var(--border);margin:8px 0;'>"
        "<div style='color:var(--cyan);margin-bottom:0.5rem;font-weight:700;'>SÍNDROMES DE FILA (XOR Paridad)</div>"
        "<table style='border-collapse:collapse;width:100%;'>"
        "<tr style='color:var(--muted);font-size:0.65rem;'>"
        "<th style='padding:3px 8px;text-align:left;'>Fila</th>"
        "<th>Paridad Esperada</th><th>Paridad Recibida</th>"
        "<th>XOR (Síndrome)</th><th>Estado</th></tr>"
    )
    for sd in syndrome_details:
        icon = "❌ ERROR" if sd["xor"] else "✅ OK"
        html_p4 += (
            f"<tr style='background:{M};border:1px solid #1e293b;'>"
            f"<td style='padding:4px 8px;color:var(--txt);'>Fila {sd['fila']}</td>"
            f"<td style='text-align:center;color:#06b6d4;'>{sd['esperado']}</td>"
            f"<td style='text-align:center;color:#8b5cf6;'>{sd['recibido']}</td>"
            f"<td style='text-align:center;color:{sd['color']};font-weight:bold;'>{sd['xor']}</td>"
            f"<td style='text-align:center;color:{sd['color']};'>{icon}</td></tr>"
        )
    html_p4 += (
        "</table>"
        "<div style='color:var(--cyan);margin:0.75rem 0 0.5rem;font-weight:700;'>SÍNDROMES DE COLUMNA (XOR Paridad)</div>"
        "<table style='border-collapse:collapse;width:100%;'>"
        "<tr style='color:var(--muted);font-size:0.65rem;'>"
        "<th style='padding:3px 8px;text-align:left;'>Columna</th>"
        "<th>Paridad Esperada</th><th>Paridad Recibida</th>"
        "<th>XOR (Síndrome)</th><th>Estado</th></tr>"
    )
    for sd in col_syn_details:
        icon = "❌ ERROR" if sd["xor"] else "✅ OK"
        html_p4 += (
            f"<tr style='background:{M};border:1px solid #1e293b;'>"
            f"<td style='padding:4px 8px;color:var(--txt);'>Col {sd['col']}</td>"
            f"<td style='text-align:center;color:#06b6d4;'>{sd['esperado']}</td>"
            f"<td style='text-align:center;color:#8b5cf6;'>{sd['recibido']}</td>"
            f"<td style='text-align:center;color:{sd['color']};font-weight:bold;'>{sd['xor']}</td>"
            f"<td style='text-align:center;color:{sd['color']};'>{icon}</td></tr>"
        )
    html_p4 += "</table>"
    if syn_row != -1 and syn_col != -1:
        html_p4 += (
            f"<div style='margin-top:0.75rem;background:rgba(239,68,68,0.1);border:1px solid #ef4444;"
            f"padding:8px 12px;border-radius:6px;color:#ef4444;'>"
            f"🎯 <strong>Error localizado en: Fila={syn_row}, Columna={syn_col}</strong></div>"
        )
    else:
        html_p4 += (
            "<div style='margin-top:0.75rem;background:rgba(16,185,129,0.1);border:1px solid #10b981;"
            "padding:8px 12px;border-radius:6px;color:#10b981;'>"
            "✅ <strong>Síndrome = 0 — Trama limpia, sin errores detectados.</strong></div>"
        )
    html_p4 += "</div>"

    # ── PASO 5: Corrección ──
    corrected_matrix = [row[:] for row in rx_matrix]
    correction_html = ""
    if syn_row != -1 and syn_col != -1:
        wrong_bit = corrected_matrix[syn_row][syn_col]
        fixed_bit = '0' if wrong_bit == '1' else '1'
        corrected_matrix[syn_row][syn_col] = fixed_bit
        correction_html = (
            "<div style='font-family:\"IBM Plex Mono\",monospace;background:var(--bg-1);"
            "padding:1rem;border-radius:8px;border:1px solid var(--border);margin:8px 0;'>"
            f"<div style='color:#10b981;font-weight:700;margin-bottom:0.5rem;'>✅ BIT CORREGIDO</div>"
            f"<div style='display:flex;align-items:center;gap:16px;flex-wrap:wrap;'>"
            f"<div style='text-align:center;'>"
            f"<div style='font-size:0.65rem;color:var(--muted);'>Posición</div>"
            f"<div style='font-size:1.2rem;color:#f59e0b;'>({syn_row}, {syn_col})</div></div>"
            f"<div style='font-size:2rem;color:var(--muted);'>→</div>"
            f"<div style='text-align:center;'>"
            f"<div style='font-size:0.65rem;color:var(--muted);'>Bit Erróneo</div>"
            f"<div style='background:#991b1b;color:white;padding:4px 14px;border-radius:6px;"
            f"font-size:1.4rem;font-weight:bold;'>{wrong_bit}</div></div>"
            f"<div style='font-size:2rem;color:#10b981;'>→</div>"
            f"<div style='text-align:center;'>"
            f"<div style='font-size:0.65rem;color:var(--muted);'>Bit Corregido</div>"
            f"<div style='background:#065f46;color:#a7f3d0;padding:4px 14px;border-radius:6px;"
            f"font-size:1.4rem;font-weight:bold;'>{fixed_bit}</div></div>"
            f"</div>"
        )
        correction_html += "<div style='margin-top:0.75rem;'>Matriz corregida:<br>"
        for r in range(num_data_rows):
            for c in range(cols):
                if r == syn_row and c == syn_col:
                    correction_html += cell(corrected_matrix[r][c], "parity-bit",
                                             "background:#065f46;border:2px solid #10b981;")
                else:
                    correction_html += cell(corrected_matrix[r][c], "data-bit")
            correction_html += cell(corrected_matrix[r][cols], "parity-bit") + "<br>"
        correction_html += "</div></div>"
    else:
        correction_html = (
            "<div style='font-family:\"IBM Plex Mono\",monospace;background:rgba(16,185,129,0.05);"
            "padding:1rem;border-radius:8px;border:1px solid #10b981;margin:8px 0;"
            "color:#10b981;'>✅ No se requiere corrección — todos los síndromes son 0.</div>"
        )

    # ── Tasas y métricas FEC ──
    n_data_bits = num_data_rows * cols
    n_parity_bits = num_data_rows + cols + 1  # paridades de fila + col + master
    n_total_bits = n_data_bits + n_parity_bits
    code_rate = n_data_bits / n_total_bits if n_total_bits > 0 else 0
    overhead_pct = (n_parity_bits / n_total_bits) * 100 if n_total_bits > 0 else 0

    # BER simulado con la señal recibida
    encoded_stream, _, _, _ = MatrixFEC(cols=cols).encode(bits)
    rx_stream = inject_bit_errors(encoded_stream, ber)
    ber_real = sum(1 for a, b in zip(encoded_stream, rx_stream) if a != b) / max(len(encoded_stream), 1)

    snr_db = -10 * math.log10(max(ber_real, 1e-9)) if ber_real > 0 else 30.0
    bits_corregidos = 1 if (syn_row != -1 and syn_col != -1) else 0

    return {
        "html_p1": html_p1,
        "html_p2": html_p2,
        "html_p3": html_p3,
        "html_p4": html_p4,
        "html_p5": correction_html,
        "err_r": err_r, "err_c": err_c,
        "syn_row": syn_row, "syn_col": syn_col,
        "code_rate": code_rate,
        "overhead_pct": overhead_pct,
        "n_data_bits": n_data_bits,
        "n_parity_bits": n_parity_bits,
        "n_total_bits": n_total_bits,
        "ber_real": ber_real,
        "snr_db": snr_db,
        "bits_corregidos": bits_corregidos,
        "encoded_stream": encoded_stream,
        "rx_stream": rx_stream,
        "padding_needed": padding_needed,
    }


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


# ═════════════════════════════════════════════════════════════════════════════
#  CANAL LAB: CORE CLASSES
# ═════════════════════════════════════════════════════════════════════════════
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


@dataclass(order=True)
class HuffNode:
    freq: int
    symbol: Any = field(compare=False)
    left: Any = field(compare=False, default=None)
    right: Any = field(compare=False, default=None)


class HuffmanCoderCanal:
    """Huffman Coder para el pipeline Tx/Rx del Lab Canal."""

    def encode(self, text: str):
        freq = Counter(text)
        heap = [HuffNode(v, k) for k, v in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            l, r = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, HuffNode(l.freq + r.freq, None, l, r))
        codes: Dict[str, str] = {}

        def gen_codes(n: HuffNode, current: str = '') -> None:
            if n.symbol is not None:
                codes[n.symbol] = current or '0'
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


class MatrixFEC:
    def __init__(self, cols: int = 4) -> None:
        self.cols = cols

    def calculate_parity(self, bit_string: str) -> str:
        return '1' if bit_string.count('1') % 2 != 0 else '0'

    def encode(self, bits: str):
        if not bits: return "", [], "", 0
        padding_needed = (self.cols - (len(bits) % self.cols)) % self.cols
        padded_bits = bits + ('0' * padding_needed)
        rows = len(padded_bits) // self.cols
        matrix, encoded_stream = [], ""
        visual_html = "<div style='margin-bottom:10px;'>"
        col_parities = ['0'] * self.cols

        for r in range(rows):
            row_data = padded_bits[r * self.cols:(r + 1) * self.cols]
            row_parity = self.calculate_parity(row_data)
            matrix.append(list(row_data) + [row_parity])
            for bit in row_data:
                visual_html += f"<span class='matrix-cell data-bit'>{bit}</span>"
            visual_html += f"<span class='matrix-cell parity-bit'>{row_parity}</span><br>"
            for c in range(self.cols):
                if row_data[c] == '1': col_parities[c] = '0' if col_parities[c] == '1' else '1'
            encoded_stream += row_data + row_parity

        master_parity = self.calculate_parity("".join(col_parities))
        for cp in col_parities:
            visual_html += f"<span class='matrix-cell parity-bit'>{cp}</span>"
        visual_html += (f"<span class='matrix-cell parity-bit' style='background-color:#9333ea;'>"
                        f"{master_parity}</span><br></div>")
        encoded_stream += "".join(col_parities) + master_parity
        return encoded_stream, matrix, visual_html, padding_needed

    def decode_and_correct(self, rx_stream: str, padding_needed: int):
        if not rx_stream: return "", "", [], "Sin datos."
        row_len = self.cols + 1
        num_data_rows = (len(rx_stream) // row_len) - 1
        rx_matrix, idx = [], 0
        for _ in range(num_data_rows + 1):
            rx_matrix.append(list(rx_stream[idx:idx + row_len]))
            idx += row_len

        error_row, error_col, syndrome_logs = -1, -1, []

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

        correction_log = "✅ Síndrome 0: Trama limpia."
        if error_row != -1 and error_col != -1:
            bad_bit = rx_matrix[error_row][error_col]
            corrected_bit = '0' if bad_bit == '1' else '1'
            correction_log = (f"🎯 **Corrección:** Coordenada ({error_row},{error_col}). "
                              f"Bit `{bad_bit}` ➔ `{corrected_bit}`.")
            rx_matrix[error_row][error_col] = corrected_bit

        visual_html = "<div>"
        for r in range(len(rx_matrix)):
            for c in range(len(rx_matrix[r])):
                is_err = (r == error_row and c == error_col)
                if is_err: css = "error-bit"
                elif c == self.cols or r == num_data_rows: css = "parity-bit"
                else: css = "data-bit"
                visual_html += f"<span class='matrix-cell {css}'>{rx_matrix[r][c]}</span>"
            visual_html += "<br>"
        visual_html += "</div>"

        clean_bits = "".join("".join(rx_matrix[r][:-1]) for r in range(num_data_rows))
        if padding_needed > 0:
            clean_bits = clean_bits[:-padding_needed]
        return clean_bits, visual_html, syndrome_logs, correction_log


def process_full_image_dct(img_arr: np.ndarray):
    h, w = img_arr.shape
    h_pad, w_pad = (8 - h % 8) % 8, (8 - w % 8) % 8
    padded = np.pad(img_arr, ((0, h_pad), (0, w_pad)), mode='constant')
    dct_blocks = np.zeros_like(padded, dtype=float)
    for i in range(0, padded.shape[0], 8):
        for j in range(0, padded.shape[1], 8):
            block = padded[i:i+8, j:j+8]
            dct_blocks[i:i+8, j:j+8] = np.round(
                scipy_dct(scipy_dct(block.T, norm='ortho').T, norm='ortho') / 10)
    return dct_blocks, padded.shape


def reconstruct_full_image_idct(dct_blocks: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    idct_blocks = np.zeros_like(dct_blocks, dtype=float)
    for i in range(0, dct_blocks.shape[0], 8):
        for j in range(0, dct_blocks.shape[1], 8):
            block = dct_blocks[i:i+8, j:j+8]
            idct_blocks[i:i+8, j:j+8] = scipy_idct(scipy_idct((block * 10).T, norm='ortho').T, norm='ortho')
    return np.clip(idct_blocks[:orig_h, :orig_w], 0, 255).astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
#  FUENTE LAB: COMPRESSION CLASSES
# ═════════════════════════════════════════════════════════════════════════════
class NodoHuffman:
    __slots__ = ("simbolo", "frecuencia", "izquierda", "derecha")

    def __init__(self, simbolo: Optional[str], frecuencia: int) -> None:
        self.simbolo, self.frecuencia = simbolo, frecuencia
        self.izquierda = self.derecha = None

    def __lt__(self, other: "NodoHuffman") -> bool:
        return self.frecuencia < other.frecuencia


class CodificadorHuffman:
    def __init__(self, datos: bytes) -> None:
        self._datos = datos

    def _construir_arbol(self, frecuencias: Dict[str, int]) -> Optional[NodoHuffman]:
        heap: List[NodoHuffman] = [NodoHuffman(s, f) for s, f in frecuencias.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            izq, der = heapq.heappop(heap), heapq.heappop(heap)
            padre = NodoHuffman(None, izq.frecuencia + der.frecuencia)
            padre.izquierda, padre.derecha = izq, der
            heapq.heappush(heap, padre)
        return heap[0] if heap else None

    def _generar_codigos(self, nodo: Optional[NodoHuffman], prefijo: str = "",
                         codigos: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        if codigos is None: codigos = {}
        if nodo is None: return codigos
        if nodo.simbolo is not None:
            codigos[nodo.simbolo] = prefijo if prefijo else "0"
        else:
            self._generar_codigos(nodo.izquierda, prefijo + "0", codigos)
            self._generar_codigos(nodo.derecha, prefijo + "1", codigos)
        return codigos

    def _generar_dot(self, nodo: Optional[NodoHuffman], total_simbolos: int) -> str:
        if not nodo: return ""
        lineas = [
            "digraph Huffman {", "    rankdir=RL;", '    bgcolor="transparent";',
            '    node [fontname="monospace", style="filled", fillcolor="#1e293b", fontcolor="#cdd9e5", color="#06b6d4"];',
            '    edge [fontname="monospace", fontcolor="#ef4444", color="#8b9cb5", fontsize=10];',
        ]
        contador = [0]

        def escapar(sym: Any) -> str:
            if isinstance(sym, int): return f"0x{sym:02X}"
            s = repr(sym)
            if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
                s = s[1:-1]
            return s.replace("\\", "\\\\").replace('"', '\\"') if s else "Espacio"

        def recorrer(n: NodoHuffman) -> str:
            my_id = f"N{contador[0]}"; contador[0] += 1
            prob = n.frecuencia / total_simbolos if total_simbolos > 0 else 0
            if n.simbolo is not None:
                lineas.append(f'    {my_id} [label="{escapar(n.simbolo)}: {prob:.2g}", '
                              f'shape="box", fillcolor="#0f172a", color="#06b6d4"];')
            else:
                lineas.append(f'    {my_id} [label="{prob:.2g}", shape="ellipse", color="#8b5cf6"];')
            if n.izquierda: lineas.append(f'    {my_id} -> {recorrer(n.izquierda)} [label=" 0 ", fontcolor="#06b6d4"];')
            if n.derecha:   lineas.append(f'    {my_id} -> {recorrer(n.derecha)}  [label=" 1 ", fontcolor="#ef4444"];')
            return my_id

        recorrer(nodo); lineas.append("}"); return "\n".join(lineas)

    def comprimir(self) -> ResultadoCompresion:
        t0 = time.perf_counter()
        try: texto = self._datos.decode("utf-8", errors="replace")
        except Exception: texto = self._datos.decode("latin-1", errors="replace")
        freq = Counter(texto)
        raiz = self._construir_arbol(dict(freq))
        codigos = self._generar_codigos(raiz)
        total_bits = sum(len(codigos.get(c, "?")) for c in texto)
        comprimido = bytes(math.ceil(total_bits / 8))
        pasos = [
            {"titulo": "Paso 1 · Frecuencias", "detalle": f"Símbolos únicos: {len(freq)}"},
            {"titulo": "Paso 2 · Árbol (Min-Heap)", "detalle": "Fusión Bottom-Up completada."},
        ]
        grafo_dot = self._generar_dot(raiz, len(texto)) if len(freq) <= 60 else None
        return ResultadoCompresion(
            "Huffman", self._datos, comprimido, self._datos,
            len(self._datos), max(1, len(comprimido)),
            len(self._datos) / max(1, len(comprimido)),
            1 - (max(1, len(comprimido)) / len(self._datos)),
            (time.perf_counter() - t0) * 1000, pasos, codigos,
            grafo_dot=grafo_dot, es_stub=True,
        )


class CodificadorLZW:
    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        t0 = time.perf_counter()
        texto = datos.decode("utf-8", errors="replace")[:10_000]
        dic, sig, w, codigos_salida = {chr(i): i for i in range(256)}, 256, "", []
        for c in texto:
            wc = w + c
            if wc in dic: w = wc
            else:
                codigos_salida.append(dic[w])
                if sig < 4096: dic[wc] = sig; sig += 1
                w = c
        if w: codigos_salida.append(dic[w])
        comprimido = struct.pack(f">{len(codigos_salida)}H", *[min(c, 65535) for c in codigos_salida]) if codigos_salida else b""
        datos_reconstruidos = b""
        if comprimido:
            codigos_lzw = struct.unpack(f">{len(comprimido)//2}H", comprimido)
            dic_dec = {i: chr(i) for i in range(256)}
            sig_dec = 256; w_dec = chr(codigos_lzw[0]); res_desc = [w_dec]
            for codigo in codigos_lzw[1:]:
                if codigo in dic_dec: entrada = dic_dec[codigo]
                elif codigo == sig_dec: entrada = w_dec + w_dec[0]
                else: break
                res_desc.append(entrada); dic_dec[sig_dec] = w_dec + entrada[0]; sig_dec += 1; w_dec = entrada
            datos_reconstruidos = "".join(res_desc).encode("utf-8", errors="replace")
        return ResultadoCompresion(
            "LZW", datos, comprimido, datos_reconstruidos, len(datos), max(1, len(comprimido)),
            len(datos) / max(1, len(comprimido)), 1 - (max(1, len(comprimido)) / len(datos)),
            (time.perf_counter() - t0) * 1000,
            [{"titulo": "Matching en Diccionario", "detalle": "LZW ejecutado correctamente."}],
            es_stub=False,
        )


class CodificadorRLE:
    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        t0 = time.perf_counter()
        if not datos:
            return ResultadoCompresion("RLE", b"", b"", b"", 0, 0, 1.0, 0.0, 0.0, es_stub=False)
        comprimido = bytearray(); runs_log = []; i = 0
        while i < len(datos):
            byte_actual = datos[i]; conteo = 1
            while i + conteo < len(datos) and datos[i + conteo] == byte_actual and conteo < 255:
                conteo += 1
            comprimido.extend([conteo, byte_actual])
            if len(runs_log) < 25: runs_log.append({"Run": conteo, "Byte": f"0x{byte_actual:02X}"})
            i += conteo
        html_rle = '<div class="rle-wrapper">'
        for run in runs_log[:18]:
            html_rle += (f'<div class="rle-pill"><div class="rle-count">{run["Run"]}×</div>'
                         f'<div class="rle-val">{run["Byte"]}</div></div>')
        html_rle += '</div>'
        decomp = bytearray(); idx = 0
        while idx + 1 < len(comprimido):
            decomp.extend([comprimido[idx + 1]] * comprimido[idx]); idx += 2
        return ResultadoCompresion(
            "RLE", datos, bytes(comprimido), bytes(decomp), len(datos), max(1, len(comprimido)),
            len(datos) / max(1, len(comprimido)), 1 - (max(1, len(comprimido)) / len(datos)),
            (time.perf_counter() - t0) * 1000,
            [{"titulo": "Bloques RLE", "detalle": "Datos agrupados:", "html": html_rle}],
            es_stub=False,
        )


class CodificadorDCT:
    def comprimir(self, datos: bytes, calidad: int = 50) -> ResultadoCompresion:
        t0 = time.perf_counter()
        raw = list(datos[:64]) + [128] * max(0, 64 - len(datos))
        html_dct = '<div class="dct-wrapper"><div class="dct-title">Bloque 8×8 Extraído</div><div class="dct-grid">'
        for val in raw[:64]:
            color = max(0, min(255, int(val))); tc = "black" if color > 127 else "white"
            html_dct += f'<div class="dct-cell" style="background-color:rgb({color},{color},{color});color:{tc};">{color}</div>'
        html_dct += '</div></div>'
        orig = len(datos); theo = max(0.05, 1 - (calidad / 100) * 0.92); comp = max(1, int(orig * theo))
        return ResultadoCompresion(
            "DCT", datos, bytes(comp), datos, orig, comp, orig / comp, 1 - (comp / orig),
            (time.perf_counter() - t0) * 1000,
            [{"titulo": "Partición en Bloques 8×8", "detalle": "Bloque extraído:", "html": html_dct}],
            es_stub=True,
        )


class CodificadorMuLaw:
    BIAS = 132

    def _encode_sample(self, sample: int) -> int:
        sample = max(-32768, min(32767, sample))
        sign = 0x00 if sample >= 0 else 0x80
        magnitude = min(abs(sample), 32635) + self.BIAS
        exp = max(0, min(int(math.log2(magnitude)) - 7, 7))
        mantissa = (magnitude >> (exp + 3)) & 0x0F
        return (~(sign | (exp << 4) | mantissa)) & 0xFF

    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        t0 = time.perf_counter()
        n_samples = len(datos) // 2
        samples = struct.unpack(f"<{n_samples}h", datos[:n_samples * 2])
        encoded = bytes(self._encode_sample(s) for s in samples)
        decoded = []
        for b in encoded:
            byte = ~b & 0xFF; sign = byte & 0x80; exp = (byte >> 4) & 0x07; mant = byte & 0x0F
            mag = ((mant << 1) | 1) << (exp + 2); val = -(mag - self.BIAS) if sign else (mag - self.BIAS)
            decoded.append(val)
        datos_rec = struct.pack(f"<{len(decoded)}h", *decoded)
        return ResultadoCompresion(
            "μ-Law", datos, encoded, datos_rec, len(datos), len(encoded), 2.0, 0.5,
            (time.perf_counter() - t0) * 1000,
            [{"titulo": "G.711 μ-Law", "detalle": "Companding logarítmico aplicado."}],
            es_stub=False,
        )


class CodificadorADPCM:
    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        comp_len = max(1, len(datos) // 4)
        return ResultadoCompresion(
            "ADPCM", datos, bytes(comp_len), datos, len(datos), comp_len, 4.0, 0.75, 1.0,
            [{"titulo": "IMA ADPCM", "detalle": "Stub educativo — ratio teórico 4:1."}], es_stub=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  SHARED UI HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def fmt_bytes(n: int) -> str:
    if n < 1024: return f"{n} B"
    if n < 1024 ** 2: return f"{n / 1024:.2f} KB"
    return f"{n / 1024 ** 2:.2f} MB"


def render_header() -> None:
    st.markdown(
        '<div class="app-header-wrap"><div class="app-header">'
        '<div class="header-icon">📡</div>'
        '<div><p class="app-title">Telecom DSP Lab · Fuente &amp; Canal</p>'
        '<p class="app-subtitle">Shannon · Huffman · LZW · RLE · DCT · μ-Law · ADPCM · FEC Matricial 2D · BPSK / QPSK / QAM16</p>'
        '</div></div></div>',
        unsafe_allow_html=True,
    )


def section_label(icon: str, text: str) -> None:
    st.markdown(
        f'<div class="section-label"><span class="sl-icon">{icon}</span>'
        f'<span class="sl-text">{text}</span></div>',
        unsafe_allow_html=True,
    )


def info_box(icon: str, content: str, variant: str = "") -> None:
    st.markdown(
        f'<div class="info-box {variant}"><div class="ib-icon">{icon}</div>'
        f'<div class="ib-content">{content}</div></div>',
        unsafe_allow_html=True,
    )


def render_dashboard(stats: EstadisticasInfo) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Entropía H(X)", f"{stats.entropia:.4f}", "bits / símbolo")
    with c2: st.metric("Long. Promedio L̄", f"{stats.longitud_promedio:.4f}", "bits / símbolo")
    with c3: st.metric("Eficiencia η", f"{stats.eficiencia * 100:.2f}%", f"Redundancia {stats.redundancia * 100:.2f}%")
    with c4: st.metric("Alfabeto", f"{stats.simbolos_unicos}", f"de {stats.total_simbolos:,} símbolos")


def render_codigo_y_formulas(algo: str, stats: EstadisticasInfo) -> None:
    section_label("📜", f"CÓDIGO Y FÓRMULAS — {algo}")
    info_box("🧮",
             f"<strong>Entropía:</strong> H(X) = −Σ p(xᵢ)·log₂p(xᵢ) = "
             f"<strong>{stats.entropia:.4f} bits/símbolo</strong>", "amber")
    st.code(_CODE_MAP.get(algo, "# Código no disponible para este algoritmo."), language="python")


def render_resultado_compresion(res: ResultadoCompresion) -> None:
    section_label("📦", f"RESULTADO — {res.nombre_algoritmo}")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Original", fmt_bytes(res.tamaño_original))
    with c2: st.metric("Comprimido", fmt_bytes(res.tamaño_comprimido))
    with c3: st.metric("Tasa", f"{res.tasa_compresion:.2f}:1")
    with c4: st.metric("Reducción", f"{max(0, res.ratio_reduccion) * 100:.1f}%")
    if res.es_stub:
        info_box("🔧", "Modo Educativo (STUB): Los bytes de salida son teóricos.", "amber")


def render_pasos(pasos: List[Dict[str, Any]]) -> None:
    section_label("🔍", "PROCEDIMIENTO PASO A PASO")
    for i, paso in enumerate(pasos, 1):
        with st.expander(paso.get("titulo", f"Paso {i}"), expanded=(i <= 2)):
            if paso.get("detalle"):
                st.markdown(
                    f"<div style='font-size:0.8rem;color:var(--txt-dim);margin-bottom:10px;white-space:pre-wrap'>"
                    f"{paso['detalle']}</div>", unsafe_allow_html=True)
            if paso.get("html"): st.markdown(paso["html"], unsafe_allow_html=True)
            elif paso.get("tabla"): st.dataframe(pd.DataFrame(paso["tabla"]), use_container_width=True, height=250)


def render_arbol_huffman(res: ResultadoCompresion) -> None:
    if res.nombre_algoritmo == "Huffman" and res.grafo_dot:
        section_label("🌿", "ÁRBOL DE HUFFMAN")
        info_box("💡", "Árbol Derecha→Izquierda con probabilidades. Azul=hoja, Morado=nodo interno.")
        st.graphviz_chart(res.grafo_dot)


def render_descodificacion(res: ResultadoCompresion, tipo_dato: str) -> None:
    section_label("🔄", "DESCODIFICACIÓN Y RECONSTRUCCIÓN")
    if res.es_stub:
        info_box("⚠️", "Algoritmo en modo STUB. Reconstrucción simulada con los datos originales.", "amber")
    else:
        info_box("✅", "Descompresión matemática real ejecutada exitosamente.", "green")

    if tipo_dato == "texto":
        try: texto_recon = res.datos_descomprimidos.decode("utf-8")
        except Exception: texto_recon = res.datos_descomprimidos.decode("latin-1", errors="replace")
        st.text_area("Texto Reconstruido",
                     value=texto_recon[:1000] + ("..." if len(texto_recon) > 1000 else ""),
                     height=150, disabled=True)
    elif tipo_dato == "imagen":
        c1, c2 = st.columns(2)
        with c1: st.markdown("**Original**"); st.image(res.datos_originales, use_container_width=True)
        with c2: st.markdown("**Reconstruida**"); st.image(res.datos_descomprimidos, use_container_width=True)
    elif tipo_dato == "audio":
        st.markdown("**Audio Reconstruido**")
        if res.nombre_algoritmo == "μ-Law":
            try:
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(44100)
                    wf.writeframes(res.datos_descomprimidos)
                st.audio(buf.getvalue(), format="audio/wav")
            except Exception: st.info("Bytes reconstruidos (no reproducible en este formato directo).")
        else:
            st.audio(res.datos_descomprimidos)
    elif tipo_dato == "video":
        c1, _ = st.columns([1, 1])
        with c1: st.markdown("**Video Reconstruido**"); st.video(res.datos_descomprimidos)


def render_no_file(icon: str, texto: str, subtexto: str) -> None:
    st.markdown(
        f'<div class="no-file-state"><div class="nfs-icon">{icon}</div>'
        f'<div class="nfs-title">{texto}</div>'
        f'<div class="nfs-sub">{subtexto}</div></div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  CANAL LAB: TAB RENDERERS
# ═════════════════════════════════════════════════════════════════════════════
def _render_fec_metricas(fv: dict) -> None:
    """Muestra métricas de canal FEC en columnas."""
    section_label("📊", "MÉTRICAS DE CANAL FEC")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Tasa de Código R", f"{fv['code_rate']:.3f}", f"k/n")
    with c2: st.metric("Overhead FEC", f"{fv['overhead_pct']:.1f}%", f"{fv['n_parity_bits']} bits paridad")
    with c3: st.metric("BER Canal", f"{fv['ber_real']:.4f}", "bits erróneos / total")
    with c4: st.metric("SNR Estimado", f"{fv['snr_db']:.1f} dB", "")
    with c5: st.metric("Bits Corregidos", str(fv['bits_corregidos']), "por FEC 2D")
    c6, c7, c8 = st.columns(3)
    with c6: st.metric("Bits de Datos", str(fv['n_data_bits']), "")
    with c7: st.metric("Bits de Paridad", str(fv['n_parity_bits']), "")
    with c8: st.metric("Bits Totales Tx", str(fv['n_total_bits']), "")


def _render_fec_paso_a_paso(fv: dict) -> None:
    """Renderiza los 5 pasos de FEC Matricial 2D con HTML visual."""
    section_label("🛡️", "FEC MATRICIAL 2D — EXPLICACIÓN PASO A PASO")
    info_box("🧠",
             "El <strong>FEC (Forward Error Correction)</strong> matricial 2D organiza los bits en una "
             "matriz rectangular. Cada fila y cada columna tiene un <strong>bit de paridad par (XOR)</strong>. "
             "Al llegar al receptor, se recalculan todas las paridades. Si una fila Y una columna "
             "tienen síndrome ≠ 0, su intersección localiza <strong>exactamente</strong> el bit erróneo "
             "y se invierte — sin retransmisión. Tasa de código: R = k/n.", "violet")

    steps = [
        ("📐 PASO 1 — MATRIZ ORIGINAL DE DATOS",
         "Los bits Huffman se ordenan en una matriz de <strong>k filas × cols columnas</strong>. "
         "Cada celda azul es un bit de información puro. "
         "Esta es la representación antes de añadir redundancia.",
         fv["html_p1"]),

        ("➕ PASO 2 — CÁLCULO DE PARIDADES (Codificación)",
         "<strong>Paridad de fila:</strong> XOR de todos los bits de cada fila → bit verde al final.<br>"
         "<strong>Paridad de columna:</strong> XOR de todos los bits de cada columna → fila verde al fondo.<br>"
         "<strong>Paridad maestra:</strong> XOR de todas las paridades de columna → celda morada (esquina).<br>"
         "Fórmula: p = b₀ ⊕ b₁ ⊕ b₂ ⊕ … ⊕ bₙ (donde ⊕ es XOR bit a bit).",
         fv["html_p2"]),

        (f"⚡ PASO 3 — ERROR INYECTADO (Canal AWGN)",
         f"El canal introduce ruido gaussiano (AWGN). Un bit en la posición "
         f"<strong>Fila {fv['err_r']}, Columna {fv['err_c']}</strong> fue invertido por el canal. "
         f"El bit rojo parpadeante es el <strong>bit erróneo recibido</strong>. "
         f"Las paridades de fila y columna de ese bit ya no coinciden con lo transmitido.",
         fv["html_p3"]),

        ("🔬 PASO 4 — DETECTOR DE SÍNDROME",
         "El receptor recalcula <strong>cada paridad</strong> y la compara (XOR) con la recibida.<br>"
         "Si el XOR de una fila = 1 → esa fila contiene un error.<br>"
         "Si el XOR de una columna = 1 → esa columna contiene un error.<br>"
         "La <strong>intersección fila×columna</strong> con síndrome=1 identifica el bit exacto.",
         fv["html_p4"]),

        ("✅ PASO 5 — CORRECCIÓN DEL BIT",
         "El receptor invierte el bit en la intersección detectada: <code>0→1</code> o <code>1→0</code>. "
         "Esto se hace <strong>sin retransmisión</strong> (por eso se llama Forward Error Correction). "
         "La tasa de corrección es de 1 error por bloque (el FEC 2D detecta 2 errores pero solo corrige 1).",
         fv["html_p5"]),
    ]

    for titulo, detalle, html in steps:
        with st.expander(titulo, expanded=True):
            st.markdown(
                f"<div style='font-size:0.8rem;color:var(--txt-dim);margin-bottom:8px;line-height:1.6;'>"
                f"{detalle}</div>",
                unsafe_allow_html=True
            )
            st.markdown(html, unsafe_allow_html=True)


def render_canal_texto() -> None:
    col_tx, col_rx = st.columns(2)

    with col_tx:
        section_label("📤", "MÓDULO TRANSMISOR")
        c1, c2, c3 = st.columns(3)
        with c1: mod_txt = st.selectbox("Modulación", ["QPSK", "BPSK", "QAM16"], key="c_tx_mod")
        with c2: fec_cols = st.selectbox("Ancho FEC", [4, 8, 16], key="c_tx_fec")
        with c3: ber_txt = st.slider("BER AWGN", 0.0, 1.0, 0.05, key="c_tx_ber")
        st.info(recommend_modulation(fec_cols))
        text_input = st.text_area("Payload:", "HOLA MUNDO", key="c_txt_payload")

        if st.button("▶ Ejecutar Pipeline Tx", type="primary", key="c_btn_tx_txt"):
            tx_bits, inverse, _, freq = HuffmanCoderCanal().encode(text_input)

            # ── Métricas de Fuente ──
            section_label("📊", "MÉTRICAS DE FUENTE (Huffman)")
            total_sym = sum(freq.values())
            probs = {s: f / total_sym for s, f in freq.items()}
            H = -sum(p * math.log2(p) for p in probs.values() if p > 0)
            L_bar = len(tx_bits) / total_sym if total_sym > 0 else 0
            eta = H / L_bar if L_bar > 0 else 0
            c1m, c2m, c3m, c4m = st.columns(4)
            with c1m: st.metric("Entropía H(X)", f"{H:.4f}", "bits/símbolo")
            with c2m: st.metric("Long. Promedio L̄", f"{L_bar:.4f}", "bits/símbolo")
            with c3m: st.metric("Eficiencia η", f"{eta*100:.2f}%", "")
            with c4m: st.metric("Redundancia", f"{(1-eta)*100:.2f}%", "")

            # ── FEC Visual completo ──
            fv = render_fec_visual_steps(tx_bits, cols=fec_cols, ber=ber_txt)
            _render_fec_paso_a_paso(fv)
            _render_fec_metricas(fv)

            fec = MatrixFEC(cols=fec_cols)
            fec_bits, _, tx_html, padding = fec.encode(tx_bits)

            with st.expander("📡 Constelación + AWGN", expanded=True):
                mod = Modulator(mod_txt)
                signal = mod.modulate(fec_bits)
                if len(signal) > 0:
                    st.pyplot(mod.render_constellation(mod.channel_awgn(signal, ber_txt)))

            meta = {"inverse": inverse, "alg": "Huffman", "cols": fec_cols, "padding": padding}
            payload = {"modulo": "texto", "metadata": meta,
                       "original_fec_bits": fec_bits,
                       "rx_bits": inject_bit_errors(fec_bits, ber_txt)}

            section_label("⬇️", "DESCARGAS")
            # Payload canal
            st.markdown(create_download_link(payload, "texto_canal.bin"), unsafe_allow_html=True)
            # Bits fuente codificados (Huffman)
            st.markdown(
                create_bytes_download_link(tx_bits.encode(), "huffman_encoded.txt",
                                           "⬇ Descargar Bits Fuente Codificados (Huffman)"),
                unsafe_allow_html=True)
            # Señal modulada como JSON
            if len(signal) > 0:
                sig_json = json.dumps({"scheme": mod_txt, "signal": signal.tolist()})
                st.markdown(
                    create_bytes_download_link(sig_json.encode(), "signal_modulada.json",
                                               "⬇ Descargar Señal Modulada"),
                    unsafe_allow_html=True)

    with col_rx:
        section_label("📥", "MÓDULO RECEPTOR")
        rx_file = st.file_uploader("Sube .bin", type=["bin", "json"], key="c_rx_txt")
        if rx_file:
            data = json.load(rx_file)
            with st.expander("🛠️ Síndromes FEC 2D", expanded=True):
                fec = MatrixFEC(cols=data["metadata"]["cols"])
                source_bits, rx_html, syndromes, correction = fec.decode_and_correct(
                    data["rx_bits"], data["metadata"]["padding"])
                ca, cb = st.columns(2)
                with ca:
                    st.markdown("**Matriz Rx:**")
                    st.markdown(rx_html, unsafe_allow_html=True)
                with cb:
                    st.markdown("**Log CPU:**")
                    for s in syndromes: st.write(s)
                    st.success(correction)
            with st.expander("🧩 Decodificación Huffman", expanded=True):
                try:
                    res_txt, logs = HuffmanCoderCanal().decode_visual_log(
                        source_bits, data["metadata"]["inverse"])
                    st.markdown(f"> **OUTPUT:** `{res_txt}`")
                    if logs: st.dataframe(pd.DataFrame(logs), use_container_width=True)
                    # Descarga del texto recuperado
                    section_label("⬇️", "DESCARGAS Rx")
                    st.markdown(
                        create_bytes_download_link(res_txt.encode(), "texto_recuperado.txt",
                                                   "⬇ Descargar Texto Recuperado"),
                        unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error de decodificación: {e}")


def render_canal_imagen() -> None:
    col_tx, col_rx = st.columns(2)

    with col_tx:
        section_label("📤", "COMPRESIÓN ESPACIAL (DCT)")
        c1, c2, c3 = st.columns(3)
        with c1: fec_img_cols = st.selectbox("Ancho FEC", [4, 8, 16], key="c_tx_img_fec")
        with c2: ber_img = st.slider("BER", 0.0, 1.0, 0.05, key="c_tx_img_ber")
        with c3: mod_img = st.selectbox("Modulación", ["QPSK", "BPSK", "QAM16"], key="c_tx_img_mod")
        img_file = st.file_uploader("Sube Imagen", type=["png", "jpg"], key="c_img_up")

        if st.button("▶ Ejecutar DCT Global", type="primary", key="c_btn_tx_img") and img_file:
            if not PIL_AVAILABLE:
                st.error("Instala Pillow: pip install Pillow")
            elif not SCIPY_AVAILABLE:
                st.error("Instala scipy: pip install scipy")
            else:
                img_raw = PILImage.open(img_file).convert("L")
                img_raw.thumbnail((64, 64))
                img_arr = np.array(img_raw)
                dct_blocks, padded_shape = process_full_image_dct(img_arr)
                tx_bits = ''.join(format(int(abs(x)), '08b') for x in dct_blocks.flatten())

                # Métricas de fuente DCT
                section_label("📊", "MÉTRICAS DE FUENTE (DCT)")
                orig_bytes = len(img_arr.flatten())
                comp_bytes = max(1, len(tx_bits) // 8)
                tasa = orig_bytes / comp_bytes
                vals = list(img_arr.flatten())
                from collections import Counter as _Ctr
                freq_im = _Ctr(vals)
                tot_im = sum(freq_im.values())
                H_im = -sum((v/tot_im)*math.log2(v/tot_im) for v in freq_im.values() if v > 0)
                ci1, ci2, ci3 = st.columns(3)
                with ci1: st.metric("Entropía H(X)", f"{H_im:.4f}", "bits/pixel")
                with ci2: st.metric("Tasa DCT", f"{tasa:.2f}:1", "")
                with ci3: st.metric("Pixels → Bits", f"{orig_bytes} → {len(tx_bits)}", "")

                # FEC visual
                fv = render_fec_visual_steps(tx_bits[:128], cols=fec_img_cols, ber=ber_img)
                _render_fec_paso_a_paso(fv)
                _render_fec_metricas(fv)

                fec = MatrixFEC(cols=fec_img_cols)
                fec_bits, _, _, padding = fec.encode(tx_bits)
                meta = {
                    "signs": np.sign(dct_blocks).flatten().tolist(),
                    "orig_h": img_arr.shape[0], "orig_w": img_arr.shape[1],
                    "pad_h": padded_shape[0], "pad_w": padded_shape[1],
                    "cols": fec_img_cols, "padding": padding,
                }
                payload = {"modulo": "imagen", "metadata": meta,
                           "original_fec_bits": fec_bits,
                           "rx_bits": inject_bit_errors(fec_bits, ber_img)}

                # Constelación
                with st.expander("📡 Constelación Imagen", expanded=False):
                    mod = Modulator(mod_img)
                    sig = mod.modulate(fec_bits[:512])
                    if len(sig) > 0:
                        st.pyplot(mod.render_constellation(mod.channel_awgn(sig, ber_img)))

                section_label("⬇️", "DESCARGAS")
                st.markdown(create_download_link(payload, "imagen_canal.bin"), unsafe_allow_html=True)
                st.markdown(
                    create_bytes_download_link(tx_bits.encode(), "dct_bits_fuente.txt",
                                               "⬇ Descargar Bits DCT Codificados"),
                    unsafe_allow_html=True)

    with col_rx:
        section_label("📥", "RECONSTRUCCIÓN IDCT")
        rx_file_img = st.file_uploader("Sube .bin", type=["bin", "json"], key="c_rx_img")
        if rx_file_img:
            data = json.load(rx_file_img)
            with st.expander("🛠️ FEC Imagen"):
                fec = MatrixFEC(cols=data["metadata"]["cols"])
                source_bits, _, syndromes, correction = fec.decode_and_correct(
                    data["rx_bits"], data["metadata"]["padding"])
                if syndromes:
                    st.error("Síndromes detectados. Reparación matricial ejecutada.")
                    st.success(correction)
                else:
                    st.success("✅ Sin errores detectados en canal imagen.")
            with st.expander("🧩 Renderizado Final", expanded=True):
                try:
                    vals = [int(source_bits[i:i+8], 2) for i in range(0, len(source_bits), 8)]
                    vals = vals[:(data["metadata"]["pad_h"] * data["metadata"]["pad_w"])]
                    quant_rx = (np.array(vals) * np.array(data["metadata"]["signs"])).reshape(
                        (data["metadata"]["pad_h"], data["metadata"]["pad_w"]))
                    rec_arr = reconstruct_full_image_idct(
                        quant_rx, data["metadata"]["orig_h"], data["metadata"]["orig_w"])
                    st.success("Reconstrucción Exitosa")
                    st.image(PILImage.fromarray(rec_arr), caption="Imagen Reconstruida en Rx",
                             use_container_width=True)
                    # Descargar imagen reconstruida
                    rec_pil = PILImage.fromarray(rec_arr)
                    buf_img = io.BytesIO()
                    rec_pil.save(buf_img, format="PNG")
                    section_label("⬇️", "DESCARGAS Rx")
                    st.markdown(
                        create_bytes_download_link(buf_img.getvalue(), "imagen_recuperada.png",
                                                   "⬇ Descargar Imagen Recuperada"),
                        unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error de reconstrucción: {e}")


def render_canal_audio() -> None:
    col_tx, col_rx = st.columns(2)

    with col_tx:
        section_label("📤", "COMPANDING μ-LAW")
        c1, c2, c3 = st.columns(3)
        with c1: fec_aud_cols = st.selectbox("Ancho FEC", [4, 8, 16], key="c_tx_aud_fec")
        with c2: ber_aud = st.slider("BER", 0.0, 1.0, 0.05, key="c_tx_aud_ber")
        with c3: mod_aud = st.selectbox("Modulación", ["BPSK", "QPSK", "QAM16"], key="c_tx_aud_mod")
        aud_file = st.file_uploader("Sube Audio WAV", type=["wav"], key="c_aud_up")

        if st.button("▶ Ejecutar Pipeline Audio", type="primary", key="c_btn_tx_aud") and aud_file:
            with wave.open(io.BytesIO(aud_file.read()), 'rb') as wav_in:
                params = wav_in.getparams()
                frames = wav_in.readframes(min(params.nframes, 16000))
            samples = np.frombuffer(frames, dtype=np.int16)
            encoded = MuLawCodec.encode(samples)
            tx_bits = ''.join(format(x & 0xFF, '08b') for x in encoded.tolist())

            # Métricas fuente
            section_label("📊", "MÉTRICAS DE FUENTE (μ-Law G.711)")
            orig_bytes = len(samples) * 2
            comp_bytes = len(encoded)
            tasa_aud = orig_bytes / max(comp_bytes, 1)
            vals_aud = list(samples.astype(np.int16))
            from collections import Counter as _Ctr2
            freq_aud = _Ctr2(np.clip(samples // 1024, -16, 15).tolist())
            tot_aud = sum(freq_aud.values())
            H_aud = -sum((v/tot_aud)*math.log2(v/tot_aud) for v in freq_aud.values() if v > 0)
            ca1, ca2, ca3, ca4 = st.columns(4)
            with ca1: st.metric("Entropía H(X)", f"{H_aud:.4f}", "bits/muestra")
            with ca2: st.metric("Tasa μ-Law", f"{tasa_aud:.2f}:1", "16-bit → 8-bit")
            with ca3: st.metric("Muestras Orig", f"{len(samples):,}", "")
            with ca4: st.metric("Bytes μ-Law", f"{comp_bytes:,}", "")

            # FEC visual
            fv = render_fec_visual_steps(tx_bits[:128], cols=fec_aud_cols, ber=ber_aud)
            _render_fec_paso_a_paso(fv)
            _render_fec_metricas(fv)

            fec = MatrixFEC(cols=fec_aud_cols)
            fec_bits, _, _, padding = fec.encode(tx_bits)

            # Constelación
            with st.expander("📡 Constelación Audio", expanded=False):
                mod = Modulator(mod_aud)
                sig = mod.modulate(fec_bits[:512])
                if len(sig) > 0:
                    st.pyplot(mod.render_constellation(mod.channel_awgn(sig, ber_aud)))

            meta = {"params": params._asdict(), "cols": fec_aud_cols, "padding": padding}
            payload = {"modulo": "audio", "metadata": meta,
                       "original_fec_bits": fec_bits,
                       "rx_bits": inject_bit_errors(fec_bits, ber_aud)}

            section_label("⬇️", "DESCARGAS")
            st.markdown(create_download_link(payload, "audio_canal.bin"), unsafe_allow_html=True)
            st.markdown(
                create_bytes_download_link(tx_bits.encode(), "mulaw_bits_fuente.txt",
                                           "⬇ Descargar Bits μ-Law Codificados"),
                unsafe_allow_html=True)

    with col_rx:
        section_label("📥", "EXPANSIÓN A PCM")
        rx_file_aud = st.file_uploader("Sube .bin", type=["bin", "json"], key="c_rx_aud")
        if rx_file_aud:
            data = json.load(rx_file_aud)
            with st.expander("🛠️ FEC Audio"):
                fec = MatrixFEC(cols=data["metadata"]["cols"])
                source_bits, _, syndromes, correction = fec.decode_and_correct(
                    data["rx_bits"], data["metadata"]["padding"])
                if syndromes: st.warning(f"Anomalías corregidas: {correction}")
                else: st.success("✅ Sin errores detectados en canal audio.")
            with st.expander("🧩 Reproducción DAC", expanded=True):
                try:
                    rx_bytes = [int(source_bits[i:i+8], 2) for i in range(0, len(source_bits), 8)]
                    decoded_pcm = MuLawCodec.decode(np.array(rx_bytes, dtype=np.uint8).astype(np.int8))
                    out_buffer = io.BytesIO()
                    p = data["metadata"]["params"]
                    with wave.open(out_buffer, 'wb') as wav_out:
                        wav_out.setparams((p['nchannels'], p['sampwidth'], p['framerate'],
                                           len(decoded_pcm) // p['nchannels'],
                                           p['comptype'], p['compname']))
                        wav_out.writeframes(decoded_pcm.tobytes())
                    st.success("Reconstrucción de Audio Exitosa")
                    st.audio(out_buffer.getvalue())
                    section_label("⬇️", "DESCARGAS Rx")
                    st.markdown(
                        create_bytes_download_link(out_buffer.getvalue(), "audio_recuperado.wav",
                                                   "⬇ Descargar Audio Recuperado WAV"),
                        unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")


def render_canal_video() -> None:
    col_tx, col_rx = st.columns(2)

    with col_tx:
        section_label("📤", "PROTECCIÓN CABECERA NAL")
        c1, c2, c3 = st.columns(3)
        with c1: fec_vid_cols = st.selectbox("Ancho FEC", [4, 8, 16], key="c_tx_vid_fec")
        with c2: ber_vid = st.slider("BER", 0.0, 1.0, 0.05, key="c_tx_vid_ber")
        with c3: mod_vid = st.selectbox("Modulación", ["QAM16", "QPSK", "BPSK"], key="c_tx_vid_mod")
        vid_file = st.file_uploader("Sube Video MP4", type=["mp4", "mov"], key="c_vid_up")
        if vid_file:
            st.markdown("**Video Original:**")
            vid_bytes = vid_file.read()
            st.video(vid_bytes)
            if st.button("▶ Transmitir Video", type="primary", key="c_btn_tx_vid"):
                header_bytes = vid_bytes[:250]
                body_bytes = vid_bytes[250:]
                tx_bits = bytes_to_bits(header_bytes)

                # Métricas fuente video
                section_label("📊", "MÉTRICAS DE FUENTE (H.264 Cabecera)")
                from collections import Counter as _Ctr3
                freq_vid = _Ctr3(list(header_bytes))
                tot_vid = sum(freq_vid.values())
                H_vid = -sum((v/tot_vid)*math.log2(v/tot_vid) for v in freq_vid.values() if v > 0)
                cv1, cv2, cv3 = st.columns(3)
                with cv1: st.metric("Entropía Cabecera", f"{H_vid:.4f}", "bits/byte")
                with cv2: st.metric("Bytes Cabecera", f"{len(header_bytes)}", "NAL Header")
                with cv3: st.metric("Bytes Cuerpo", f"{len(body_bytes):,}", "Stream H.264")

                # FEC visual
                fv = render_fec_visual_steps(tx_bits[:128], cols=fec_vid_cols, ber=ber_vid)
                _render_fec_paso_a_paso(fv)
                _render_fec_metricas(fv)

                fec = MatrixFEC(cols=fec_vid_cols)
                fec_bits, _, tx_html, padding = fec.encode(tx_bits)

                with st.expander("🛡️ FEC Cabecera MP4 (Matriz)", expanded=False):
                    st.caption("Protegiendo metadata del contenedor (Atoms):")
                    st.markdown(tx_html, unsafe_allow_html=True)

                # Constelación
                with st.expander("📡 Constelación Video", expanded=False):
                    mod = Modulator(mod_vid)
                    sig = mod.modulate(fec_bits[:512])
                    if len(sig) > 0:
                        st.pyplot(mod.render_constellation(mod.channel_awgn(sig, ber_vid)))

                meta = {"body_b64": base64.b64encode(body_bytes).decode('utf-8'),
                        "cols": fec_vid_cols, "padding": padding}
                payload = {"modulo": "video", "metadata": meta,
                           "original_fec_bits": fec_bits,
                           "rx_bits": inject_bit_errors(fec_bits, ber_vid)}
                section_label("⬇️", "DESCARGAS")
                st.markdown(create_download_link(payload, "video_canal.bin"), unsafe_allow_html=True)
                st.markdown(
                    create_bytes_download_link(tx_bits.encode(), "nal_header_bits.txt",
                                               "⬇ Descargar Bits Cabecera NAL"),
                    unsafe_allow_html=True)

    with col_rx:
        section_label("📥", "RECEPCIÓN H.264")
        rx_file_vid = st.file_uploader("Sube .bin", type=["bin", "json"], key="c_rx_vid")
        if rx_file_vid:
            data = json.load(rx_file_vid)
            with st.expander("🛠️ Reparación Cabecera"):
                fec = MatrixFEC(cols=data["metadata"]["cols"])
                source_bits, _, syndromes, correction = fec.decode_and_correct(
                    data["rx_bits"], data["metadata"]["padding"])
                if syndromes:
                    st.warning("Síndromes no nulos. Reparando Atoms MP4...")
                    st.success(correction)
                else:
                    st.success("✅ Cabecera intacta — sin errores en canal video.")
            try:
                source_bits = source_bits[:len(source_bits) - (len(source_bits) % 8)]
                header_rx = bits_to_bytes(source_bits)
                body_rx = base64.b64decode(data["metadata"]["body_b64"])
                full_video = header_rx + body_rx
                st.success("Reconstrucción Exitosa: Cabecera NAL intacta.")
                st.markdown("**Video Recibido:**")
                st.video(full_video)
                section_label("⬇️", "DESCARGAS Rx")
                st.markdown(
                    create_bytes_download_link(full_video, "video_recuperado.mp4",
                                               "⬇ Descargar Video Recuperado MP4"),
                    unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Video corrupto — ruido destruyó la estructura: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  FUENTE LAB: TAB RENDERERS
# ═════════════════════════════════════════════════════════════════════════════
def render_fuente_texto() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")
    with col_up:
        section_label("📄", "CARGAR ARCHIVO DE TEXTO")
        uploaded = st.file_uploader("Sube un archivo de texto", type=["txt", "csv", "json"], key="f_up_txt")
    with col_info:
        section_label("ℹ️", "ALGORITMOS DISPONIBLES")
        st.info("Huffman · LZW · RLE")

    if uploaded is None:
        render_no_file("📝", "Sube un archivo de texto", ".txt .csv .json")
        return

    datos = uploaded.read()
    stats = AnalizadorTexto(datos, uploaded.name).calcular_todo()
    render_dashboard(stats)
    section_label("⚙️", "ALGORITMO")
    c_sel, c_btn = st.columns([3, 1])
    with c_sel: algo = st.selectbox("Algoritmo", ["Huffman", "LZW", "RLE"], key="f_algo_txt", label_visibility="collapsed")
    with c_btn: btn = st.button("▶ Comprimir", key="f_btn_txt")
    if btn:
        render_codigo_y_formulas(algo, stats)
        if algo == "Huffman": res = CodificadorHuffman(datos).comprimir()
        elif algo == "LZW": res = CodificadorLZW().comprimir(datos)
        else: res = CodificadorRLE().comprimir(datos)
        render_resultado_compresion(res); render_pasos(res.pasos)
        render_arbol_huffman(res); render_descodificacion(res, "texto")
        section_label("⬇️", "DESCARGAS — FUENTE CODIFICADA")
        st.markdown(
            create_bytes_download_link(res.datos_comprimidos,
                                       f"fuente_{algo.lower()}_codificado.bin",
                                       f"⬇ Descargar Fuente Codificada ({algo})"),
            unsafe_allow_html=True)


def render_fuente_imagen() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")
    with col_up:
        section_label("🖼️", "CARGAR IMAGEN")
        uploaded = st.file_uploader("Sube una imagen", type=["png", "jpg"], key="f_up_img")
    with col_info:
        section_label("ℹ️", "ALGORITMOS DISPONIBLES")
        st.info("DCT (JPEG) · RLE · Huffman")

    if uploaded is None:
        render_no_file("🖼️", "Sube una imagen", ".png .jpg")
        return

    datos = uploaded.read()
    st.image(datos, width=250)
    stats = AnalizadorImagen(datos, uploaded.name).calcular_todo()
    render_dashboard(stats)
    section_label("⚙️", "ALGORITMO")
    c_sel, c_btn = st.columns([3, 1])
    with c_sel: algo = st.selectbox("Algoritmo", ["DCT — JPEG-like", "RLE", "Huffman (bytes)"], key="f_algo_img", label_visibility="collapsed")
    with c_btn: btn = st.button("▶ Comprimir", key="f_btn_img")
    if btn:
        llave = "Huffman" if algo == "Huffman (bytes)" else algo
        render_codigo_y_formulas(llave, stats)
        if algo == "DCT — JPEG-like": res = CodificadorDCT().comprimir(datos)
        elif algo == "RLE": res = CodificadorRLE().comprimir(datos)
        else: res = CodificadorHuffman(datos).comprimir()
        render_resultado_compresion(res); render_pasos(res.pasos)
        render_arbol_huffman(res); render_descodificacion(res, "imagen")
        section_label("⬇️", "DESCARGAS — FUENTE CODIFICADA")
        st.markdown(
            create_bytes_download_link(res.datos_comprimidos,
                                       f"imagen_{llave.lower().replace(' ','_')}_codificada.bin",
                                       f"⬇ Descargar Imagen Codificada ({llave})"),
            unsafe_allow_html=True)


def render_fuente_audio() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")
    with col_up:
        section_label("🎵", "CARGAR AUDIO")
        uploaded = st.file_uploader("Sube audio (.wav)", type=["wav"], key="f_up_aud")
    with col_info:
        section_label("ℹ️", "ALGORITMOS DISPONIBLES")
        st.info("μ-Law G.711 · ADPCM · Huffman")

    if not uploaded:
        render_no_file("🎵", "Sube un archivo de audio", ".wav")
        return

    datos = uploaded.read()
    st.audio(datos)
    stats = AnalizadorAudio(datos, uploaded.name).calcular_todo()
    render_dashboard(stats)
    section_label("⚙️", "ALGORITMO")
    c_sel, c_btn = st.columns([3, 1])
    with c_sel: algo = st.selectbox("Algoritmo", ["μ-Law G.711", "ADPCM", "Huffman (bytes)"], key="f_algo_aud", label_visibility="collapsed")
    with c_btn: btn = st.button("▶ Comprimir", key="f_btn_aud")
    if btn:
        llave = "Huffman" if algo == "Huffman (bytes)" else algo
        render_codigo_y_formulas(llave, stats)
        if algo == "μ-Law G.711": res = CodificadorMuLaw().comprimir(datos)
        elif algo == "ADPCM": res = CodificadorADPCM().comprimir(datos)
        else: res = CodificadorHuffman(datos).comprimir()
        render_resultado_compresion(res); render_pasos(res.pasos)
        render_arbol_huffman(res); render_descodificacion(res, "audio")
        section_label("⬇️", "DESCARGAS — FUENTE CODIFICADA")
        st.markdown(
            create_bytes_download_link(res.datos_comprimidos,
                                       f"audio_{llave.lower().replace(' ','_').replace('.','')}_codificado.bin",
                                       f"⬇ Descargar Audio Codificado ({llave})"),
            unsafe_allow_html=True)


def render_fuente_video() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")
    with col_up:
        section_label("🎬", "CARGAR VIDEO")
        uploaded = st.file_uploader("Sube un video", type=["mp4", "avi", "mkv", "mov", "webm"], key="f_up_vid")
    with col_info:
        section_label("ℹ️", "CONCEPTOS APLICADOS")
        st.info("Tramas I/P/B · DCT Temporal · Estimación Movimiento · CABAC")

    if uploaded is None:
        render_no_file("🎬", "Sube un archivo de video", ".mp4 .avi .mkv .mov .webm")
        return

    datos = uploaded.read()
    v1, _ = st.columns([1, 1])
    with v1: st.video(datos)
    with st.spinner("Analizando flujo de bytes de video..."):
        stats = AnalizadorVideo(datos, uploaded.name).calcular_todo()
    render_dashboard(stats)
    info_box("🔬",
             f"El flujo H.264/HEVC ya comprimido tiene entropía alta "
             f"(<strong>{stats.entropia:.4f} bits/byte</strong>) porque sus datos ya están "
             f"codificados entrópicamente. Un video RAW tendría entropía menor.", "violet")

    section_label("⚙️", "SIMULACIÓN H.264")
    _, c_btn = st.columns([3, 1])
    with c_btn: btn = st.button("▶ Simular H.264", key="f_btn_vid")

    if btn:
        render_codigo_y_formulas("H.264 / HEVC (Simulado)", stats)

        html_mb = (
            '<div style="display:flex;flex-direction:column;align-items:center;background:var(--bg-1);'
            'padding:1rem;border-radius:8px;border:1px solid var(--border);margin:1rem 0;">'
            '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;color:var(--cyan);'
            'margin-bottom:1rem;">FRAME DIVIDIDO EN MACROBLOCKS 16×16</div>'
            '<div style="display:grid;grid-template-columns:repeat(8,1fr);gap:2px;">' +
            ''.join('<div style="width:20px;height:20px;background:var(--bg-2);'
                    'border:1px solid rgba(6,182,212,0.3);"></div>' for _ in range(64)) +
            '</div></div>'
        )

        html_motion = (
            '<div style="display:flex;flex-wrap:wrap;gap:20px;align-items:center;justify-content:center;'
            'background:var(--bg-1);padding:1.5rem;border-radius:8px;border:1px solid var(--border);margin:1rem 0;">'
            '<div style="text-align:center;">'
            '<div style="width:80px;height:80px;border:2px dashed var(--muted);position:relative;background:var(--bg-2);">'
            '<div style="width:25px;height:25px;background:var(--violet);position:absolute;top:10px;left:10px;border-radius:4px;"></div>'
            '</div><div style="font-size:0.65rem;color:var(--muted);margin-top:8px;font-family:monospace;">Frame Ref (t-1)</div></div>'
            '<div style="display:flex;flex-direction:column;align-items:center;color:var(--cyan);">'
            '<div style="font-weight:bold;font-family:monospace;font-size:0.8rem;background:rgba(6,182,212,0.1);'
            'padding:4px 8px;border-radius:4px;border:1px solid rgba(6,182,212,0.3);">MV (dx:+20, dy:+15)</div>'
            '<div style="font-size:1.5rem;margin-top:4px;">➔</div></div>'
            '<div style="text-align:center;">'
            '<div style="width:80px;height:80px;border:2px solid var(--border-strong);position:relative;background:var(--bg-2);">'
            '<div style="width:25px;height:25px;background:var(--cyan);position:absolute;top:25px;left:30px;'
            'border-radius:4px;box-shadow:0 0 10px rgba(6,182,212,0.5);"></div>'
            '<div style="width:25px;height:25px;border:2px dashed rgba(139,92,246,0.5);position:absolute;'
            'top:10px;left:10px;border-radius:4px;"></div>'
            '</div><div style="font-size:0.65rem;color:var(--cyan);margin-top:8px;font-family:monospace;">Frame Actual (t)</div>'
            '</div></div>'
        )

        res_vals = [5, -2, 0, 1, 0, 0, 0, 0, -3, 1, 0, 0, 0, 0, 0, 0] + [0] * 48
        html_residual = (
            '<div class="dct-wrapper"><div class="dct-title">DCT Residual (errores ~ 0)</div>'
            '<div class="dct-grid">' +
            ''.join(
                f'<div class="dct-cell" style="background-color:rgb({128+v*10},{128+v*10},{128+v*10});'
                f'color:{"#ef4444" if v != 0 else "var(--muted)"};border:1px solid var(--bg-0);">{v}</div>'
                for v in res_vals
            ) + '</div></div>'
        )

        html_cabac = (
            '<div class="rle-wrapper">'
            '<div class="rle-pill"><div class="rle-count" style="background:var(--bg-4);color:var(--txt-dim);">Símbolos</div>'
            '<div class="rle-val">0,1,0,0...</div></div>'
            '<span style="color:var(--muted);padding:0 8px;font-size:1.2rem;">➔</span>'
            '<div class="rle-pill"><div class="rle-count" style="background:linear-gradient(135deg,var(--amber),#d97706);">'
            'Modelo Ctx</div></div>'
            '<span style="color:var(--muted);padding:0 8px;font-size:1.2rem;">➔</span>'
            '<div class="rle-pill"><div class="rle-count" style="background:linear-gradient(135deg,var(--green),#059669);">CABAC</div>'
            '<div class="rle-val" style="color:var(--green);font-weight:bold;">01101...</div></div></div>'
        )

        pasos = [
            {"titulo": "Paso 1 · Partición en Macroblocks", "detalle": "División de cada frame en bloques 16×16.", "html": html_mb},
            {"titulo": "Paso 2 · Estimación de Movimiento", "detalle": "Vectores de movimiento para tramas P y B.", "html": html_motion},
            {"titulo": "Paso 3 · DCT + Cuantización", "detalle": "DCT sobre los residuales y reducción de precisión.", "html": html_residual},
            {"titulo": "Paso 4 · CABAC", "detalle": "Codificación aritmética adaptiva basada en contexto.", "html": html_cabac},
        ]
        orig = len(datos) * 45
        res = ResultadoCompresion(
            "H.264 (Simulado)", datos, datos, datos, orig, len(datos),
            orig / len(datos), 1 - (len(datos) / orig), 125.4, pasos, es_stub=True)
        render_resultado_compresion(res)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Trama I (Intra)", "3.5:1", "Solo DCT espacial")
        with c2: st.metric("Trama P (Predictiva)", "12.0:1", "DCT + Motion Vector")
        with c3: st.metric("Trama B (Bidireccional)", "20.0:1", "Máxima compresión")
        render_pasos(res.pasos)
        render_descodificacion(res, "video")
        section_label("⬇️", "DESCARGAS — FUENTE CODIFICADA")
        st.markdown(
            create_bytes_download_link(res.datos_comprimidos,
                                       "video_h264_sim_codificado.bin",
                                       "⬇ Descargar Stream H.264 Simulado"),
            unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    inject_css()
    render_header()

    lab_tabs = st.tabs([
        "📊 Lab Fuente · Compresión & Entropía",
        "📡 Lab Canal · FEC & Modulación",
    ])

    # ── LAB FUENTE ──────────────────────────────────────────────────────────
    with lab_tabs[0]:
        info_box("📚",
                 "Análisis de fuentes de información y algoritmos de compresión: "
                 "<strong>Huffman</strong>, <strong>LZW</strong>, <strong>RLE</strong>, "
                 "<strong>DCT/JPEG</strong>, <strong>μ-Law G.711</strong>, <strong>ADPCM</strong>, "
                 "<strong>H.264</strong> simulado. Incluye métricas de Entropía, "
                 "Longitud Promedio y Eficiencia.", "violet")
        sub_f = st.tabs(["📄 Texto", "🖼️ Imagen", "🎵 Audio", "🎬 Video"])
        with sub_f[0]: render_fuente_texto()
        with sub_f[1]: render_fuente_imagen()
        with sub_f[2]: render_fuente_audio()
        with sub_f[3]: render_fuente_video()

    # ── LAB CANAL ───────────────────────────────────────────────────────────
    with lab_tabs[1]:
        info_box("📡",
                 "Pipeline completo <strong>Tx / Rx</strong>: "
                 "Huffman → <strong>FEC Matricial 2D</strong> (Paridad Cruzada) → "
                 "Modulación <strong>BPSK / QPSK / QAM16</strong> → "
                 "Canal <strong>AWGN</strong> → FEC Decode → Reconstrucción. "
                 "Descarga el payload .BIN en Tx y súbelo en Rx.")
        sub_c = st.tabs(["📄 Texto", "🖼️ Imagen Real", "🎵 Audio Físico", "🎬 Video H.264"])
        with sub_c[0]: render_canal_texto()
        with sub_c[1]: render_canal_imagen()
        with sub_c[2]: render_canal_audio()
        with sub_c[3]: render_canal_video()


if __name__ == "__main__":
    main()
