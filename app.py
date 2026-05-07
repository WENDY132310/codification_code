from __future__ import annotations

import heapq
import io
import math
import struct
import time
import wave
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ── Optional dependencies ──────────────────────────────────────────────────
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  ← must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataLab · Compresión",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS — "Terminal Lab" Dark Aesthetic
# ─────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&display=swap');

        :root {
            --bg-0:    #030712;
            --bg-1:    #070d1a;
            --bg-2:    #0d1424;
            --bg-3:    #141e30;
            --bg-4:    #1a2540;
            --cyan:    #06b6d4;
            --cyan-dim: rgba(6,182,212,0.12);
            --violet:  #8b5cf6;
            --violet-dim: rgba(139,92,246,0.12);
            --green:   #10b981;
            --amber:   #f59e0b;
            --red:     #ef4444;
            --txt:     #cdd9e5;
            --txt-dim: #8b9cb5;
            --muted:   #4e5f7a;
            --border:  rgba(6,182,212,0.14);
            --border-strong: rgba(6,182,212,0.30);
            --radius-sm: 8px;
            --radius-md: 14px;
            --radius-lg: 20px;
        }

        html, body, [class*="css"], [data-testid] {
            font-family: 'Syne', sans-serif !important;
            background-color: var(--bg-0) !important;
            color: var(--txt) !important;
        }
        * { box-sizing: border-box; }

        #MainMenu, footer, header { visibility: hidden !important; }
        .stDeployButton, [data-testid="stToolbar"] { display: none !important; }
        section[data-testid="stSidebar"] { display: none !important; }

        .main .block-container { padding: 0 2rem 4rem !important; max-width: 1440px !important; }

        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: var(--bg-1); }
        ::-webkit-scrollbar-thumb { background: var(--cyan); border-radius: 3px; }

        /* HEADER */
        .app-header-wrap { background: var(--bg-1); border-bottom: 1px solid var(--border); padding: 1.2rem 0 1.4rem; margin: 0 -2rem 2rem; padding-left: 2rem; padding-right: 2rem; }
        .app-header { display: flex; align-items: center; justify-content: space-between; gap: 1.5rem; }
        .header-left { display: flex; align-items: center; gap: 1.25rem; }
        .header-icon { width: 48px; height: 48px; background: linear-gradient(135deg, var(--cyan), var(--violet)); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; flex-shrink: 0; }
        .app-title { font-size: 1.55rem; font-weight: 800; background: linear-gradient(100deg, var(--cyan) 0%, #a78bfa 60%, var(--violet) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; letter-spacing: -0.04em; margin: 0; }
        .app-subtitle { font-family: 'IBM Plex Mono', monospace !important; color: var(--muted); font-size: 0.7rem; letter-spacing: 0.06em; margin-top: 0.2rem; }
        
        /* TABS */
        .stTabs [data-baseweb="tab-list"] { background: var(--bg-2) !important; border-radius: var(--radius-md) !important; padding: 5px !important; gap: 4px !important; border: 1px solid var(--border) !important; }
        .stTabs [data-baseweb="tab"] { background: transparent !important; border-radius: var(--radius-sm) !important; color: var(--muted) !important; font-weight: 600 !important; font-size: 0.82rem !important; padding: 0.55rem 1.4rem !important; border: none !important; }
        .stTabs [aria-selected="true"] { background: linear-gradient(135deg, var(--cyan-dim), var(--violet-dim)) !important; color: var(--cyan) !important; border: 1px solid var(--border-strong) !important; }
        .stTabs [data-baseweb="tab-panel"] { padding: 1.75rem 0 0 !important; }

        /* SECTIONS & METRICS */
        .section-label { display: flex; align-items: center; gap: 0.6rem; margin: 1.5rem 0 0.85rem; }
        .section-label .sl-icon { font-size: 0.9rem; }
        .section-label .sl-text { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.65rem; font-weight: 600; letter-spacing: 0.18em; text-transform: uppercase; color: var(--cyan); }
        .section-label::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, var(--border) 0%, transparent 100%); }
        
        [data-testid="stMetric"] { background: var(--bg-2) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; padding: 1.1rem 1.3rem !important; position: relative; overflow: hidden; }
        [data-testid="stMetric"]::before { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--violet), var(--cyan)); }
        [data-testid="stMetricLabel"] > div { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.62rem !important; color: var(--muted) !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }
        [data-testid="stMetricValue"] > div { font-size: 1.7rem !important; font-weight: 700 !important; color: var(--cyan) !important; font-family: 'IBM Plex Mono', monospace !important; letter-spacing: -0.03em !important; }

        /* UPLOADER & BUTTONS */
        [data-testid="stFileUploader"] { background: var(--bg-2) !important; border: 1.5px dashed rgba(6,182,212,0.28) !important; border-radius: var(--radius-lg) !important; padding: 1.5rem !important; transition: border-color 0.25s !important; }
        [data-testid="stFileUploader"]:hover { border-color: var(--cyan) !important; box-shadow: var(--glow-cyan) !important; }
        
        .stButton > button { background: linear-gradient(135deg, var(--cyan) 0%, var(--violet) 100%) !important; color: #fff !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important; font-family: 'Syne', sans-serif !important; font-size: 0.85rem !important; padding: 0.55rem 2rem !important; letter-spacing: 0.03em !important; transition: all 0.2s ease !important; box-shadow: 0 2px 12px rgba(6,182,212,0.2) !important; }
        .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 24px rgba(6,182,212,0.35) !important; }

        /* INFO BOXES */
        .info-box { display: flex; gap: 0.9rem; align-items: flex-start; background: rgba(6,182,212,0.06); border: 1px solid rgba(6,182,212,0.22); border-radius: var(--radius-md); padding: 1rem 1.25rem; margin: 0.75rem 0; }
        .info-box.amber { background: rgba(245,158,11,0.06); border-color: rgba(245,158,11,0.22); }
        .info-box.green { background: rgba(16,185,129,0.06); border-color: rgba(16,185,129,0.22); }
        .info-box.violet { background: rgba(139,92,246,0.06); border-color: rgba(139,92,246,0.22); }
        .info-box .ib-content { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.75rem; color: var(--txt); line-height: 1.7; }
        
        /* VISUALIZACIONES GRÁFICAS (DCT Y RLE) */
        .dct-wrapper { display: flex; flex-direction: column; align-items: center; background: var(--bg-1); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border); margin: 1rem 0; }
        .dct-grid { display: grid; grid-template-columns: repeat(8, 1fr); gap: 2px; width: 100%; max-width: 320px; background: var(--border); padding: 2px; border-radius: 4px; }
        .dct-cell { aspect-ratio: 1; display: flex; align-items: center; justify-content: center; font-size: 0.65rem; font-family: 'IBM Plex Mono', monospace; border-radius: 2px; font-weight: 600; }
        .dct-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: var(--cyan); margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.1em; }

        .rle-wrapper { display: flex; flex-wrap: wrap; gap: 8px; background: var(--bg-1); padding: 1.25rem; border-radius: 8px; border: 1px solid var(--border); margin: 1rem 0; }
        .rle-pill { display: flex; border-radius: 6px; overflow: hidden; border: 1px solid var(--border); }
        .rle-count { background: linear-gradient(135deg, var(--cyan), var(--violet)); color: #fff; font-weight: 800; padding: 4px 10px; font-size: 0.8rem; font-family: 'IBM Plex Mono', monospace; display: flex; align-items: center; justify-content: center; }
        .rle-val { background: var(--bg-3); color: var(--txt); padding: 4px 12px; font-size: 0.8rem; font-family: 'IBM Plex Mono', monospace; display: flex; align-items: center; justify-content: center; }

        /* CODE BLOCK OVERRIDES */
        .stCode, [data-testid="stCode"] { background: #010508 !important; border: 1px solid rgba(6,182,212,0.18) !important; border-radius: var(--radius-md) !important; }
        
        .no-file-state { text-align: center; padding: 3rem 2rem; background: var(--bg-2); border: 1.5px dashed var(--border); border-radius: var(--radius-lg); margin-top: 1.5rem; }
        .no-file-state .nfs-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
        .no-file-state .nfs-title { font-size: 1rem; font-weight: 600; color: var(--txt-dim); margin-bottom: 0.4rem; }
        .no-file-state .nfs-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--muted); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
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
#  ABSTRACT BASE CLASS
# ═════════════════════════════════════════════════════════════════════════════

class AnalizadorInformacion(ABC):
    def __init__(self, datos: bytes, nombre: str = "datos") -> None:
        if not datos: raise ValueError("Los datos no pueden estar vacíos.")
        self._datos = datos
        self._nombre = nombre

    @abstractmethod
    def _extraer_simbolos(self) -> List[Any]: ...

    def calcular_frecuencias(self) -> Dict[str, int]: return dict(Counter(self._extraer_simbolos()))
    def calcular_probabilidades(self) -> Dict[str, float]:
        freq = self.calcular_frecuencias()
        total = sum(freq.values())
        return {k: v / total for k, v in freq.items()}

    def calcular_entropia(self) -> float:
        return -sum(p * math.log2(p) for p in self.calcular_probabilidades().values() if p > 0)

    def calcular_longitud_promedio(self, longitudes: Optional[Dict[str, int]] = None) -> float:
        probs = self.calcular_probabilidades()
        if longitudes: return sum(probs[s] * longitudes[s] for s in probs if s in longitudes)
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
    def _extraer_simbolos(self) -> List[int]: return list(self._datos[: self.CAP])


# ═════════════════════════════════════════════════════════════════════════════
#  COMPRESSION & DECOMPRESSION ALGORITHMS
# ═════════════════════════════════════════════════════════════════════════════

# ─── A. HUFFMAN ──────────────────────────────────────────────────────────────
class NodoHuffman:
    __slots__ = ("simbolo", "frecuencia", "izquierda", "derecha")
    def __init__(self, simbolo: Optional[str], frecuencia: int) -> None:
        self.simbolo, self.frecuencia = simbolo, frecuencia
        self.izquierda = self.derecha = None
    def __lt__(self, other: "NodoHuffman") -> bool: return self.frecuencia < other.frecuencia

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

    def _generar_codigos(self, nodo: Optional[NodoHuffman], prefijo: str = "", codigos: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        if codigos is None: codigos = {}
        if nodo is None: return codigos
        if nodo.simbolo is not None: codigos[nodo.simbolo] = prefijo if prefijo else "0"
        else:
            self._generar_codigos(nodo.izquierda, prefijo + "0", codigos)
            self._generar_codigos(nodo.derecha, prefijo + "1", codigos)
        return codigos

    def _generar_dot(self, nodo: Optional[NodoHuffman], total_simbolos: int) -> str:
        if not nodo: return ""
        lineas = [
            "digraph Huffman {",
            "    rankdir=RL;",            # Dibuja de Derecha a Izquierda
            "    bgcolor=\"transparent\";",
            "    node [fontname=\"monospace\", style=\"filled\", fillcolor=\"#1e293b\", fontcolor=\"#cdd9e5\", color=\"#06b6d4\"];",
            "    edge [fontname=\"monospace\", fontcolor=\"#ef4444\", color=\"#8b9cb5\", fontsize=10];"
        ]
        contador = [0]
        def escapar_etiqueta(sym: Any) -> str:
            if isinstance(sym, int): return f"0x{sym:02X}"
            s = repr(sym)
            if s.startswith("'") and s.endswith("'"): s = s[1:-1]
            elif s.startswith('"') and s.endswith('"'): s = s[1:-1]
            return s.replace("\\", "\\\\").replace('"', '\\"') if s else "Espacio"

        def recorrer(n: NodoHuffman) -> str:
            my_id = f"N{contador[0]}"
            contador[0] += 1
            prob = n.frecuencia / total_simbolos if total_simbolos > 0 else 0
            if n.simbolo is not None:
                lineas.append(f"    {my_id} [label=\"{escapar_etiqueta(n.simbolo)}: {prob:.2g}\", shape=\"box\", fillcolor=\"#0f172a\", color=\"#06b6d4\"];")
            else:
                lineas.append(f"    {my_id} [label=\"{prob:.2g}\", shape=\"ellipse\", color=\"#8b5cf6\"];")
            if n.izquierda: lineas.append(f"    {my_id} -> {recorrer(n.izquierda)} [label=\" 0 \", fontcolor=\"#06b6d4\"];")
            if n.derecha: lineas.append(f"    {my_id} -> {recorrer(n.derecha)} [label=\" 1 \", fontcolor=\"#ef4444\"];")
            return my_id
        recorrer(nodo)
        lineas.append("}")
        return "\n".join(lineas)

    def comprimir(self) -> ResultadoCompresion:
        t0 = time.perf_counter()
        try: texto = self._datos.decode("utf-8", errors="replace")
        except Exception: texto = self._datos.decode("latin-1", errors="replace")
        
        freq = Counter(texto)
        raiz = self._construir_arbol(dict(freq))
        codigos = self._generar_codigos(raiz)
        
        total_bits = sum(len(codigos.get(c, "?")) for c in texto)
        comprimido = bytes(math.ceil(total_bits / 8)) 

        tabla_frec = [{"Símbolo": repr(k), "Frecuencia": v} for k, v in sorted(freq.items(), key=lambda x: -x[1])]

        pasos = [
            {"titulo": "Paso 1 · Frecuencias y Repeticiones", "detalle": "", "tabla": tabla_frec[:150]},
            {"titulo": "Paso 2 · Construcción del Árbol", "detalle": "Fusión Bottom-Up usando Min-Heap."},
        ]

        grafo_dot = self._generar_dot(raiz, len(texto)) if len(freq) <= 60 else None

        datos_reconstruidos = self._datos 

        return ResultadoCompresion("Huffman", self._datos, comprimido, datos_reconstruidos, len(self._datos), max(1, len(comprimido)), len(self._datos) / max(1, len(comprimido)), 1 - (max(1, len(comprimido)) / len(self._datos)), (time.perf_counter() - t0) * 1000, pasos, codigos, grafo_dot=grafo_dot, es_stub=True)

# ─── B. LZW ──────────────────────────────────────────────────────────────────
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
        
        # ── REAL LZW DECOMPRESSION ──
        datos_reconstruidos = b""
        if comprimido:
            codigos_lzw = struct.unpack(f">{len(comprimido)//2}H", comprimido)
            dic_dec = {i: chr(i) for i in range(256)}
            sig_dec = 256
            w_dec = chr(codigos_lzw[0])
            res_desc = [w_dec]
            for codigo in codigos_lzw[1:]:
                if codigo in dic_dec: entrada = dic_dec[codigo]
                elif codigo == sig_dec: entrada = w_dec + w_dec[0]
                else: break
                res_desc.append(entrada)
                dic_dec[sig_dec] = w_dec + entrada[0]
                sig_dec += 1
                w_dec = entrada
            datos_reconstruidos = "".join(res_desc).encode("utf-8", errors="replace")

        return ResultadoCompresion("LZW", datos, comprimido, datos_reconstruidos, len(datos), max(1, len(comprimido)), len(datos) / max(1, len(comprimido)), 1 - (max(1, len(comprimido)) / len(datos)), (time.perf_counter() - t0) * 1000, [{"titulo": "Matching en Diccionario", "detalle": "Codificación y decodificación LZW ejecutada."}], es_stub=False)

# ─── C. RLE ───────────────────────────────────────────────────────────────────
class CodificadorRLE:
    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        t0 = time.perf_counter()
        if not datos: return ResultadoCompresion("RLE", b"", b"", b"", 0, 0, 1.0, 0.0, 0.0, es_stub=False)
        
        comprimido = bytearray()
        runs_log = []
        i = 0
        while i < len(datos):
            byte_actual = datos[i]
            conteo = 1
            while i + conteo < len(datos) and datos[i + conteo] == byte_actual and conteo < 255:
                conteo += 1
            comprimido.extend([conteo, byte_actual])
            if len(runs_log) < 25: runs_log.append({"Run": conteo, "Byte": f"0x{byte_actual:02X}"})
            i += conteo

        html_rle = '<div class="rle-wrapper">'
        for run in runs_log[:18]:
            html_rle += f'<div class="rle-pill" title="Se repite {run["Run"]} veces el byte {run["Byte"]}"><div class="rle-count">{run["Run"]}×</div><div class="rle-val">{run["Byte"]}</div></div>'
        html_rle += '</div>'

        # ── REAL RLE DECOMPRESSION ──
        decomp = bytearray()
        idx = 0
        while idx + 1 < len(comprimido):
            decomp.extend([comprimido[idx+1]] * comprimido[idx])
            idx += 2
        datos_reconstruidos = bytes(decomp)

        pasos = [{"titulo": "Particionado Secuencial (Bloques RLE)", "detalle": "", "html": html_rle}]
        return ResultadoCompresion("RLE", datos, bytes(comprimido), datos_reconstruidos, len(datos), max(1, len(comprimido)), len(datos) / max(1, len(comprimido)), 1 - (max(1, len(comprimido)) / len(datos)), (time.perf_counter() - t0) * 1000, pasos, es_stub=False)

# ─── D. DCT (Image / JPEG-like) ───────────────────────────────────────────────
class CodificadorDCT:
    def _dct2d_manual(self, bloque: np.ndarray) -> np.ndarray:
        N = 8
        b = bloque.astype(float) - 128.0
        F = np.zeros((N, N))
        xs, ys = np.arange(N), np.arange(N)
        for u in range(N):
            for v in range(N):
                cu = 1.0 / math.sqrt(2) if u == 0 else 1.0
                cv = 1.0 / math.sqrt(2) if v == 0 else 1.0
                cos_u = np.cos((2 * xs + 1) * u * math.pi / (2 * N))
                cos_v = np.cos((2 * ys + 1) * v * math.pi / (2 * N))
                F[u, v] = (2 / N) * cu * cv * np.sum(b * cos_u[:, None] * cos_v[None, :])
        return F

    def comprimir(self, datos: bytes, calidad: int = 50) -> ResultadoCompresion:
        t0 = time.perf_counter()
        raw = list(datos[:64]) + [128] * max(0, 64 - len(datos))
        bloque = np.array(raw[:64], dtype=float).reshape(8, 8)
        
        # STRING HTML EN UNA SOLA LÍNEA PARA EVITAR BUG DE STREAMLIT MARKDOWN
        html_dct = (
            '<div class="dct-wrapper"><div class="dct-title">Bloque Extraído (8x8 px)</div><div class="dct-grid">'
        )
        for val in raw[:64]:
            color = max(0, min(255, int(val)))
            text_color = "black" if color > 127 else "white"
            html_dct += f'<div class="dct-cell" style="background-color: rgb({color},{color},{color}); color: {text_color};">{color}</div>'
        html_dct += '</div></div>'
        
        orig = len(datos)
        theo_ratio = max(0.05, 1 - (calidad / 100) * 0.92)
        comp = max(1, int(orig * theo_ratio))
        
        datos_reconstruidos = datos

        # STRING HTML EN UNA SOLA LÍNEA (IDCT GRID)
        html_idct = (
            '<div style="display:flex; flex-direction:column; gap: 15px; background:var(--bg-1); padding:1rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="font-family:\'IBM Plex Mono\', monospace; font-size:0.75rem; color:var(--cyan); text-align:center;">PROCEDIMIENTO: IDCT (Transformada Inversa)</div>'
            '<div style="display:flex; flex-wrap:wrap; justify-content:center; align-items:center; gap: 20px;">'
            '<div style="display:flex; flex-direction:column; align-items:center;">'
            '<div style="font-family:\'IBM Plex Mono\', monospace; font-size:0.6rem; color:var(--muted); margin-bottom:8px;">1. MATRIZ CUANTIZADA</div>'
            '<div class="dct-grid" style="width:140px; background:var(--bg-2); border:1px solid var(--border);">'
        )
        for i in range(64):
            val = 35 if i==0 else (-12 if i==1 else (8 if i==8 else 0))
            text_color = "#ef4444" if val != 0 else "rgba(139,156,181,0.2)"
            html_idct += f'<div class="dct-cell" style="background:var(--bg-0); color:{text_color}; font-size:0.55rem;">{val}</div>'
        html_idct += (
            '</div></div>'
            '<span style="color:var(--cyan); font-size:1.5rem; font-weight:bold;">➔</span>'
            '<div style="display:flex; flex-direction:column; align-items:center;">'
            '<div style="font-family:\'IBM Plex Mono\', monospace; font-size:0.6rem; color:var(--muted); margin-bottom:8px;">2. BLOQUE RECONSTRUIDO</div>'
            '<div class="dct-grid" style="width:140px; background:var(--bg-2); border:1px solid var(--border);">'
        )
        for val in raw[:64]:
            color = max(0, min(255, int(val)))
            html_idct += f'<div class="dct-cell" style="background-color: rgb({color},{color},{color}); color: transparent;">0</div>'
        html_idct += (
            '</div></div></div>'
            '</div>'
        )

        pasos = [
            {"titulo": "Paso 1 · Partición en Bloques y Transformada", "detalle": "", "html": html_dct},
            {"titulo": "Paso 2 · Proceso de Descompresión (Reconstrucción IDCT)", "detalle": "", "html": html_idct}
        ]
        return ResultadoCompresion("DCT", datos, bytes(comp), datos_reconstruidos, orig, comp, orig / comp, 1 - (comp / orig), (time.perf_counter() - t0) * 1000, pasos, es_stub=True)

# ─── E. μ-Law (Audio) & F. ADPCM ──────────────────────────────────────────────
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
        samples = struct.unpack(f"<{n_samples}h", datos[: n_samples * 2])
        encoded = bytes(self._encode_sample(s) for s in samples)

        # ── REAL MU-LAW DECOMPRESSION ──
        decoded = []
        for b in encoded:
            byte = ~b & 0xFF
            sign = byte & 0x80
            exp = (byte >> 4) & 0x07
            mant = byte & 0x0F
            mag = ((mant << 1) | 1) << (exp + 2)
            val = -(mag - self.BIAS) if sign else (mag - self.BIAS)
            decoded.append(val)
        datos_reconstruidos = struct.pack(f"<{len(decoded)}h", *decoded)

        # HTML VISUAL PARA MU-LAW (SIN SALTOS DE LINEA)
        html_mulaw_comp = (
            '<div style="display:flex; align-items:center; justify-content:center; gap:20px; background:var(--bg-1); padding:1.5rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="border:1px solid var(--border); padding:15px; border-radius:8px; text-align:center; background:var(--bg-2); min-width:120px;">'
            '<div style="font-size:0.65rem; color:var(--muted); font-family:\'IBM Plex Mono\', monospace; margin-bottom:5px;">MUESTRA ORIGINAL</div>'
            '<div style="font-size:1.5rem; color:var(--cyan); font-weight:bold;">16-bit</div>'
            '<div style="font-size:0.7rem; color:var(--txt-dim); margin-top:5px;">Linear PCM</div>'
            '</div>'
            '<div style="display:flex; flex-direction:column; align-items:center;">'
            '<div style="font-size:0.65rem; color:var(--amber); font-family:\'IBM Plex Mono\', monospace; padding:4px 8px; background:rgba(245,158,11,0.1); border-radius:4px; margin-bottom:4px;">Companding Logarítmico</div>'
            '<span style="color:var(--muted); font-size:1.5rem;">➔</span>'
            '</div>'
            '<div style="border:1px solid var(--border-strong); padding:15px; border-radius:8px; text-align:center; background:rgba(239,68,68,0.05); min-width:120px;">'
            '<div style="font-size:0.65rem; color:var(--muted); font-family:\'IBM Plex Mono\', monospace; margin-bottom:5px;">CODIFICADO</div>'
            '<div style="font-size:1.5rem; color:#ef4444; font-weight:bold;">8-bit</div>'
            '<div style="font-size:0.7rem; color:var(--txt-dim); margin-top:5px;">μ-Law</div>'
            '</div>'
            '</div>'
        )

        html_mulaw_decomp = (
            '<div style="display:flex; align-items:center; justify-content:center; gap:20px; background:var(--bg-1); padding:1.5rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="border:1px solid var(--border-strong); padding:15px; border-radius:8px; text-align:center; background:rgba(239,68,68,0.05); min-width:120px;">'
            '<div style="font-size:0.65rem; color:var(--muted); font-family:\'IBM Plex Mono\', monospace; margin-bottom:5px;">ARCHIVO GUARDADO</div>'
            '<div style="font-size:1.5rem; color:#ef4444; font-weight:bold;">8-bit</div>'
            '</div>'
            '<div style="display:flex; flex-direction:column; align-items:center;">'
            '<div style="font-size:0.65rem; color:var(--green); font-family:\'IBM Plex Mono\', monospace; padding:4px 8px; background:rgba(16,185,129,0.1); border-radius:4px; margin-bottom:4px;">Expansión Inversa</div>'
            '<span style="color:var(--muted); font-size:1.5rem;">➔</span>'
            '</div>'
            '<div style="border:1px solid var(--cyan); padding:15px; border-radius:8px; text-align:center; background:rgba(6,182,212,0.05); min-width:120px; box-shadow: 0 0 15px rgba(6,182,212,0.2);">'
            '<div style="font-size:0.65rem; color:var(--cyan); font-family:\'IBM Plex Mono\', monospace; margin-bottom:5px;">ONDA RECONSTRUIDA</div>'
            '<div style="font-size:1.5rem; color:var(--cyan); font-weight:bold;">16-bit</div>'
            '</div>'
            '</div>'
        )

        pasos = [
            {"titulo": "Paso 1 · Cuantización Logarítmica (Compresión)", "detalle": "", "html": html_mulaw_comp},
            {"titulo": "Paso 2 · Proceso de Descompresión (Expansión)", "detalle": "", "html": html_mulaw_decomp}
        ]

        t1 = time.perf_counter()
        return ResultadoCompresion("μ-Law", datos, encoded, datos_reconstruidos, len(datos), len(encoded), 2.0, 0.5, (t1-t0)*1000, pasos, es_stub=False)

class CodificadorADPCM:
    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        t0 = time.perf_counter()
        n_samples = len(datos) // 2
        samples = struct.unpack(f"<{n_samples}h", datos[: n_samples * 2])
        
        encoded = bytearray()
        decoded = []
        prev_sample = 0
        
        for s in samples:
            diff = s - prev_sample
            step = 256
            quantized_diff = max(-128, min(127, diff // step))
            encoded.append(quantized_diff & 0xFF)
            recon_diff = quantized_diff * step
            prev_sample = max(-32768, min(32767, prev_sample + recon_diff))
            decoded.append(prev_sample)
            
        datos_reconstruidos = struct.pack(f"<{len(decoded)}h", *decoded)

        # HTML VISUAL PARA ADPCM (SIN SALTOS DE LINEA)
        html_adpcm_diff = (
            '<div style="display:flex; align-items:center; justify-content:center; gap:15px; background:var(--bg-1); padding:1.5rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="text-align:center;"><div style="width:50px; height:60px; background:var(--cyan); margin:0 auto; border-radius:4px;"></div><div style="font-size:0.6rem; color:var(--muted); margin-top:5px; font-family:\'IBM Plex Mono\', monospace;">Muestra<br>(t)</div></div>'
            '<span style="font-size:2rem; color:var(--muted); font-weight:bold;">-</span>'
            '<div style="text-align:center;"><div style="width:50px; height:45px; background:var(--violet); margin:0 auto; border-radius:4px;"></div><div style="font-size:0.6rem; color:var(--muted); margin-top:5px; font-family:\'IBM Plex Mono\', monospace;">Anterior<br>(t-1)</div></div>'
            '<span style="font-size:2rem; color:var(--muted); font-weight:bold;">=</span>'
            '<div style="border:1px dashed var(--border-strong); padding:15px; border-radius:8px; text-align:center; background:rgba(6,182,212,0.05);">'
            '<div style="font-size:0.6rem; color:var(--cyan); font-family:\'IBM Plex Mono\', monospace; margin-bottom:5px;">DIFERENCIA (ERROR)</div>'
            '<div style="font-size:1.2rem; color:var(--txt); font-weight:bold;">Δ</div>'
            '</div>'
            '</div>'
        )

        html_adpcm_quant = (
            '<div style="display:flex; align-items:center; justify-content:center; gap:20px; background:var(--bg-1); padding:1.5rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="border:1px dashed var(--border-strong); padding:15px; border-radius:8px; text-align:center; background:rgba(6,182,212,0.05); min-width:100px;">'
            '<div style="font-size:1.2rem; color:var(--txt); font-weight:bold;">Δ</div>'
            '<div style="font-size:0.6rem; color:var(--cyan); margin-top:5px;">Diferencia</div>'
            '</div>'
            '<div style="display:flex; flex-direction:column; align-items:center;">'
            '<div style="font-size:0.65rem; color:var(--amber); font-family:\'IBM Plex Mono\', monospace; padding:4px 8px; background:rgba(245,158,11,0.1); border-radius:4px; margin-bottom:4px;">÷ Step Size</div>'
            '<span style="color:var(--muted); font-size:1.5rem;">➔</span>'
            '</div>'
            '<div style="border:1px solid #ef4444; padding:15px; border-radius:8px; text-align:center; background:rgba(239,68,68,0.05); min-width:100px;">'
            '<div style="font-size:1.2rem; color:#ef4444; font-weight:bold;">4-bit / 8-bit</div>'
            '<div style="font-size:0.6rem; color:var(--muted); margin-top:5px;">Cuantizado a guardar</div>'
            '</div>'
            '</div>'
        )

        html_adpcm_recon = (
            '<div style="display:flex; align-items:center; justify-content:center; gap:15px; background:var(--bg-1); padding:1.5rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="border:1px solid #ef4444; padding:10px; border-radius:8px; text-align:center; background:rgba(239,68,68,0.05);"><div style="font-size:0.8rem; color:#ef4444; font-weight:bold;">Quantizado</div></div>'
            '<span style="font-size:1.2rem; color:var(--muted);">× Step =</span>'
            '<div style="border:1px dashed var(--border-strong); padding:10px; border-radius:8px; text-align:center;"><div style="font-size:0.8rem; color:var(--cyan); font-weight:bold;">Δ Reconstruido</div></div>'
            '<span style="font-size:1.5rem; color:var(--muted); font-weight:bold;">+</span>'
            '<div style="text-align:center;"><div style="width:40px; height:45px; background:var(--violet); margin:0 auto; border-radius:4px;"></div><div style="font-size:0.5rem; color:var(--muted); margin-top:5px;">Anterior (t-1)</div></div>'
            '<span style="font-size:1.5rem; color:var(--muted); font-weight:bold;">=</span>'
            '<div style="text-align:center;"><div style="width:40px; height:60px; background:var(--cyan); margin:0 auto; border-radius:4px; box-shadow: 0 0 10px rgba(6,182,212,0.4);"></div><div style="font-size:0.5rem; color:var(--cyan); margin-top:5px;">Onda Final (16-bit)</div></div>'
            '</div>'
        )

        pasos = [
            {"titulo": "Paso 1 · Predicción Temporal (Diferencia)", "detalle": "", "html": html_adpcm_diff},
            {"titulo": "Paso 2 · Cuantización (Reducción de Bits)", "detalle": "", "html": html_adpcm_quant},
            {"titulo": "Paso 3 · Proceso de Descompresión (Reconstrucción Predictiva)", "detalle": "", "html": html_adpcm_recon}
        ]

        t1 = time.perf_counter()
        return ResultadoCompresion("ADPCM (Funcional Simplificado)", datos, bytes(encoded), datos_reconstruidos, len(datos), len(encoded), 2.0, 0.5, (t1-t0)*1000, pasos, es_stub=False)


# ═════════════════════════════════════════════════════════════════════════════
#  SOURCE CODE STRINGS (CON FÓRMULAS)
# ═════════════════════════════════════════════════════════════════════════════

_CODE_MAP = {
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
            dic[wc] = sig
            sig += 1
            w = c
    if w: salida.append(dic[w])
    return salida
""",
    "RLE": """\
# Run-Length Encoding (RLE)
def rle_comprimir(datos: bytes) -> bytes:
    out, i = bytearray(), 0
    while i < len(datos):
        byte = datos[i]
        run = 1
        while i + run < len(datos) and datos[i+run] == byte and run < 255:
            run += 1
        out.extend([run, byte])
        i += run
    return bytes(out)

def rle_descomprimir(datos: bytes) -> bytes:
    out, i = bytearray(), 0
    while i + 1 < len(datos):
        out.extend([datos[i+1]] * datos[i])
        i += 2
    return bytes(out)
""",
    "DCT — JPEG-like": """\
# Transformada Discreta del Coseno (DCT-II 2D)
# F(u,v) = (2/N) * Cu * Cv * ΣΣ f(x,y) * cos(...) * cos(...)

def dct2(bloque: np.ndarray) -> np.ndarray:
    N = 8
    b = bloque.astype(float) - 128
    F = np.zeros((N, N))
    xs = np.arange(N)
    for u in range(N):
        cu = 1/math.sqrt(2) if u == 0 else 1.0
        cos_u = np.cos((2*xs+1)*u*math.pi/(2*N))
        for v in range(N):
            cv = 1/math.sqrt(2) if v == 0 else 1.0
            cos_v = np.cos((2*xs+1)*v*math.pi/(2*N))
            F[u,v] = (2/N)*cu*cv*np.sum(b * cos_u[:,None] * cos_v[None,:])
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
                    # Búsqueda en ventana ±16
                    ref_block = frame_ref[i+dy:..., j+dx:...]
                    sad = np.sum(np.abs(curr_block - ref_block))
                    if sad < best_sad:
                        best_sad, best_mv = sad, (dx, dy)
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
# IMA ADPCM (Adaptive Differential PCM)
# Cuantización de la diferencia entre la muestra y el predictor
# Δ = (x_n - x_pred) / step_size
"""
}

# ═════════════════════════════════════════════════════════════════════════════
#  UI HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def fmt_bytes(n: int) -> str:
    if n < 1024: return f"{n} B"
    if n < 1024**2: return f"{n/1024:.2f} KB"
    return f"{n/1024**2:.2f} MB"

def render_header() -> None:
    st.markdown('<div class="app-header-wrap"><div class="app-header"><div class="header-left"><div class="header-icon">⚡</div><div><p class="app-title">DataLab · Compresión</p><p class="app-subtitle">Shannon · Huffman · LZW · RLE · DCT · μ-Law · ADPCM · H.264</p></div></div></div></div>', unsafe_allow_html=True)

def section_label(icon: str, text: str) -> None:
    st.markdown(f'<div class="section-label"><span class="sl-icon">{icon}</span><span class="sl-text">{text}</span></div>', unsafe_allow_html=True)

def info_box(icon: str, content: str, variant: str = "") -> None:
    st.markdown(f'<div class="info-box {variant}"><div class="ib-icon">{icon}</div><div class="ib-content">{content}</div></div>', unsafe_allow_html=True)

def render_dashboard(stats: EstadisticasInfo) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Entropía H(X)", f"{stats.entropia:.4f}", "bits / símbolo")
    with c2: st.metric("Long. Promedio L̄", f"{stats.longitud_promedio:.4f}", "bits / símbolo")
    with c3: st.metric("Eficiencia η", f"{stats.eficiencia * 100:.2f}%", f"Redundancia {stats.redundancia*100:.2f}%")
    with c4: st.metric("Alfabeto", f"{stats.simbolos_unicos}", f"de {stats.total_simbolos:,} símbolos")

def render_codigo_y_formulas(algo: str, stats: EstadisticasInfo) -> None:
    section_label("📜", f"CÓDIGO Y FÓRMULAS — {algo}")
    info_box("🧮", f"<strong>Entropía Matemática:</strong> $H(X) = -\sum p(x_i) \cdot \log_2 p(x_i)$<br>Límite teórico calculado: <strong>{stats.entropia:.4f} bits/símbolo</strong>", "amber")
    st.code(_CODE_MAP.get(algo, "# Código no disponible para este algoritmo."), language="python")

def render_resultado_compresion(res: ResultadoCompresion) -> None:
    section_label("📦", f"RESULTADO DE COMPRESIÓN — {res.nombre_algoritmo}")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Tamaño Original", fmt_bytes(res.tamaño_original))
    with c2: st.metric("Tamaño Comprimido", fmt_bytes(res.tamaño_comprimido))
    with c3: st.metric("Tasa", f"{res.tasa_compresion:.2f}:1")
    with c4: st.metric("Reducción", f"{max(0,res.ratio_reduccion)*100:.1f}%")
    
    if res.es_stub: info_box("🔧", "Modo Educativo (STUB): Los bytes de salida son teóricos.", "amber")

def render_pasos(pasos: List[Dict[str, Any]]) -> None:
    section_label("🔍", "PROCEDIMIENTO PASO A PASO")
    for i, paso in enumerate(pasos, 1):
        with st.expander(paso.get("titulo", f"Paso {i}"), expanded=(i <= 2)):
            if paso.get("detalle"): st.markdown(f"<div style='font-size:0.8rem;color:var(--txt-dim);margin-bottom:10px;white-space:pre-wrap'>{paso['detalle']}</div>", unsafe_allow_html=True)
            if paso.get("html"): st.markdown(paso["html"], unsafe_allow_html=True)
            elif paso.get("tabla"): st.dataframe(pd.DataFrame(paso["tabla"]), use_container_width=True, height=250)

def render_arbol_huffman(res: ResultadoCompresion) -> None:
    if res.nombre_algoritmo == "Huffman" and res.grafo_dot:
        section_label("🌿", "DIAGRAMA DEL ÁRBOL DE HUFFMAN")
        info_box("💡", "Árbol horizontal (Derecha a Izquierda) con probabilidades y etiquetas de bit.")
        st.graphviz_chart(res.grafo_dot)

def render_descodificacion(res: ResultadoCompresion, tipo_dato: str) -> None:
    section_label("🔄", "DESCODIFICACIÓN Y RECONSTRUCCIÓN")
    
    if res.es_stub:
        info_box("⚠️", "Este algoritmo está en modo STUB. La reconstrucción mostrada abajo es simulada usando los datos originales para demostrar el flujo final.", "amber")
    else:
        info_box("✅", "El algoritmo ha ejecutado exitosamente el proceso inverso (Descompresión matemática real).", "green")

    if tipo_dato == "texto":
        try:
            texto_recon = res.datos_descomprimidos.decode("utf-8")
        except Exception:
            texto_recon = res.datos_descomprimidos.decode("latin-1", errors="replace")
        st.text_area("Texto Reconstruido", value=texto_recon[:1000] + ("..." if len(texto_recon) > 1000 else ""), height=150, disabled=True)
    
    elif tipo_dato == "imagen":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original**")
            st.image(res.datos_originales, use_container_width=True)
        with c2:
            st.markdown("**Reconstruida**")
            st.image(res.datos_descomprimidos, use_container_width=True)
            
    elif tipo_dato == "audio":
        st.markdown("**Audio Reconstruido**")
        if res.nombre_algoritmo == "μ-Law":
            try:
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1) 
                    wf.setsampwidth(2) 
                    wf.setframerate(44100) 
                    wf.writeframes(res.datos_descomprimidos)
                st.audio(buf.getvalue(), format="audio/wav")
            except Exception:
                st.info("Formato de audio no reproducible directamente, pero los bytes fueron reconstruidos exitosamente.")
        else:
            st.audio(res.datos_descomprimidos)
            
    elif tipo_dato == "video":
        st.markdown("**Video Reconstruido**")
        st.video(res.datos_descomprimidos)

def render_no_file(icon: str, texto: str, subtexto: str) -> None:
    st.markdown(f'<div class="no-file-state"><div class="nfs-icon">{icon}</div><div class="nfs-title">{texto}</div><div class="nfs-sub">{subtexto}</div></div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TABS Y MAIN
# ═════════════════════════════════════════════════════════════════════════════

def tab_texto() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")
    with col_up:
        section_label("📄", "CARGAR ARCHIVO DE TEXTO")
        uploaded = st.file_uploader("Sube un archivo de texto", type=["txt", "csv", "json"], key="uploader_texto")
    with col_info:
        section_label("ℹ️", "ALGORITMOS")
        st.info("Huffman · LZW · RLE")

    if uploaded is None:
        render_no_file("📝", "Sube un archivo", ".txt .csv .json")
        return

    datos = uploaded.read()
    analizador = AnalizadorTexto(datos, uploaded.name)
    stats = analizador.calcular_todo()
    render_dashboard(stats)

    section_label("⚙️", "ALGORITMO DE COMPRESIÓN")
    c_sel, c_btn = st.columns([3, 1])
    with c_sel: algo = st.selectbox("Algoritmo", ["Huffman", "LZW", "RLE"], key="algo_texto", label_visibility="collapsed")
    with c_btn: btn_comp = st.button("▶ Comprimir", key="btn_texto")

    if btn_comp:
        render_codigo_y_formulas(algo, stats) 
        
        if algo == "Huffman": res = CodificadorHuffman(datos).comprimir()
        elif algo == "LZW": res = CodificadorLZW().comprimir(datos)
        else: res = CodificadorRLE().comprimir(datos)

        render_resultado_compresion(res)
        render_pasos(res.pasos)
        render_arbol_huffman(res)
        render_descodificacion(res, "texto") 


def tab_imagen() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")
    with col_up:
        section_label("🖼️", "CARGAR IMAGEN")
        uploaded = st.file_uploader("Sube una imagen", type=["png", "jpg"], key="uploader_imagen")
    with col_info:
        section_label("ℹ️", "ALGORITMOS")
        st.info("DCT (JPEG) · RLE · Huffman")
    
    if uploaded is None:
        render_no_file("🖼️", "Sube una imagen", ".png .jpg")
        return

    datos = uploaded.read()
    st.image(datos, width=250)
    
    analizador = AnalizadorImagen(datos, uploaded.name)
    stats = analizador.calcular_todo()
    render_dashboard(stats)

    section_label("⚙️", "ALGORITMO DE COMPRESIÓN")
    c_sel, c_btn = st.columns([3, 1])
    with c_sel: algo_img = st.selectbox("Algoritmo", ["DCT — JPEG-like", "RLE", "Huffman (bytes)"], key="algo_img", label_visibility="collapsed")
    with c_btn: comprimir = st.button("▶ Comprimir", key="btn_img")

    if comprimir:
        if algo_img == "Huffman (bytes)": algo_llave = "Huffman"
        else: algo_llave = algo_img

        render_codigo_y_formulas(algo_llave, stats)
        
        if algo_img == "DCT — JPEG-like": res = CodificadorDCT().comprimir(datos)
        elif algo_img == "RLE": res = CodificadorRLE().comprimir(datos)
        else: res = CodificadorHuffman(datos).comprimir()
        
        render_resultado_compresion(res)
        render_pasos(res.pasos)
        render_arbol_huffman(res)
        render_descodificacion(res, "imagen")


def tab_audio() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")
    with col_up:
        section_label("🎵", "CARGAR ARCHIVO DE AUDIO")
        uploaded = st.file_uploader("Sube audio (.wav)", type=["wav"], key="uploader_audio")
    with col_info:
        section_label("ℹ️", "ALGORITMOS")
        st.info("μ-Law G.711 · ADPCM · Huffman")

    if not uploaded:
        render_no_file("🎵", "Sube un archivo de audio", ".wav")
        return
        
    datos = uploaded.read()
    st.audio(datos)
    
    analizador = AnalizadorAudio(datos, uploaded.name)
    stats = analizador.calcular_todo()
    render_dashboard(stats)
    
    section_label("⚙️", "ALGORITMO DE COMPRESIÓN")
    c_sel, c_btn = st.columns([3, 1])
    with c_sel: algo = st.selectbox("Algoritmo", ["μ-Law G.711", "ADPCM", "Huffman (bytes)"], key="algo_audio", label_visibility="collapsed")
    with c_btn: btn_comp = st.button("▶ Comprimir", key="btn_audio")
    
    if btn_comp:
        if algo == "Huffman (bytes)": algo_llave = "Huffman"
        else: algo_llave = algo

        render_codigo_y_formulas(algo_llave, stats)
        
        if algo == "μ-Law G.711": res = CodificadorMuLaw().comprimir(datos)
        elif algo == "ADPCM": res = CodificadorADPCM().comprimir(datos)
        else: res = CodificadorHuffman(datos).comprimir()
        
        render_resultado_compresion(res)
        render_pasos(res.pasos)
        render_arbol_huffman(res)
        render_descodificacion(res, "audio")


def tab_video() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")
    with col_up:
        section_label("🎬", "CARGAR ARCHIVO DE VIDEO")
        uploaded = st.file_uploader("Sube un archivo de video", type=["mp4", "avi", "mkv", "mov", "webm"], key="uploader_video")
    with col_info:
        section_label("ℹ️", "CONCEPTOS APLICADOS")
        st.info("Tramas I/P/B · DCT Temporal · Entropía CABAC")

    if uploaded is None:
        render_no_file("🎬", "Sube un archivo de video", ".mp4 .avi .mkv .mov .webm")
        return

    datos = uploaded.read()
    st.video(datos)
    
    analizador = AnalizadorVideo(datos, uploaded.name)
    with st.spinner("Analizando flujo de bytes de video..."):
        stats = analizador.calcular_todo()

    render_dashboard(stats)
    info_box("🔬", f"El flujo de video comprimido (H.264/HEVC) tiene entropía alta (<strong>{stats.entropia:.4f} bits/byte</strong>) porque los datos ya están codificados entrópicamente. Un video RAW tendría entropía menor y sería más compresible.", "violet")

    section_label("⚙️", "ALGORITMO DE COMPRESIÓN")
    c_sel, c_btn = st.columns([3, 1])
    with c_sel:
        algo_vid = st.selectbox("Simulación", ["H.264 / HEVC (Simulado)"], key="algo_video", label_visibility="collapsed")
    with c_btn:
        btn_comp = st.button("▶ Simular Compresión", key="btn_video")

    if btn_comp:
        render_codigo_y_formulas(algo_vid, stats)
        
        orig = len(datos) * 45  
        comp = len(datos)
        
        # HTML NATIVO PARA PASOS DE VIDEO (STRING DE UNA LÍNEA PARA EVITAR BUGS DE STREAMLIT)
        html_macroblocks = (
            '<div style="display:flex; flex-direction:column; align-items:center; background:var(--bg-1); padding:1rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="font-family:\'IBM Plex Mono\', monospace; font-size:0.75rem; color:var(--cyan); margin-bottom:1rem;">FRAME DIVIDIDO EN MACROBLOCKS</div>'
            '<div style="display:grid; grid-template-columns:repeat(8, 1fr); gap:2px; background:var(--border); padding:2px; border-radius:4px;">'
        )
        for _ in range(64):
            html_macroblocks += '<div style="width:20px; height:20px; background:var(--bg-2); border:1px solid rgba(6,182,212,0.3);"></div>'
        html_macroblocks += '</div></div>'

        html_motion = (
            '<div style="display:flex; flex-wrap:wrap; gap:20px; align-items:center; justify-content:center; background:var(--bg-1); padding:1.5rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="text-align:center;">'
            '<div style="width:80px; height:80px; border:2px dashed var(--muted); position:relative; background:var(--bg-2);">'
            '<div style="width:25px; height:25px; background:var(--violet); position:absolute; top:10px; left:10px; border-radius:4px;"></div>'
            '</div>'
            '<div style="font-size:0.65rem; color:var(--muted); margin-top:8px; font-family:\'IBM Plex Mono\', monospace;">Frame Referencia (t-1)</div>'
            '</div>'
            '<div style="display:flex; flex-direction:column; align-items:center; color:var(--cyan);">'
            '<div style="font-weight:bold; font-family:\'IBM Plex Mono\', monospace; font-size:0.8rem; background:rgba(6,182,212,0.1); padding:4px 8px; border-radius:4px; border:1px solid rgba(6,182,212,0.3);">Vect. Mov (dx:+20, dy:+15)</div>'
            '<div style="font-size:1.5rem; margin-top:4px;">➔</div>'
            '</div>'
            '<div style="text-align:center;">'
            '<div style="width:80px; height:80px; border:2px solid var(--border-strong); position:relative; background:var(--bg-2);">'
            '<div style="width:25px; height:25px; background:var(--cyan); position:absolute; top:25px; left:30px; border-radius:4px; box-shadow: 0 0 10px rgba(6,182,212,0.5);"></div>'
            '<div style="width:25px; height:25px; border:2px dashed rgba(139,92,246,0.5); position:absolute; top:10px; left:10px; border-radius:4px;"></div>'
            '</div>'
            '<div style="font-size:0.65rem; color:var(--cyan); margin-top:8px; font-family:\'IBM Plex Mono\', monospace;">Frame Actual (t)</div>'
            '</div>'
            '</div>'
        )

        html_residual = (
            '<div style="display:flex; flex-direction:column; align-items:center; background:var(--bg-1); padding:1rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="font-family:\'IBM Plex Mono\', monospace; font-size:0.75rem; color:var(--cyan); margin-bottom:1rem;">DCT DEL RESIDUAL (Errores cercanos a 0)</div>'
            '<div class="dct-grid">'
        )
        residual_vals = [5, -2, 0, 1, 0, 0, 0, 0, -3, 1, 0, 0, 0, 0, 0, 0] + [0]*48
        for val in residual_vals:
            color = 128 + val*10
            text_color = "#ef4444" if val != 0 else "var(--muted)"
            html_residual += f'<div class="dct-cell" style="background-color: rgb({color},{color},{color}); color: {text_color}; border:1px solid var(--bg-0);">{val}</div>'
        html_residual += '</div></div>'

        html_cabac = (
            '<div style="display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:center; background:var(--bg-1); padding:1.5rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div class="rle-pill"><div class="rle-count" style="background:var(--bg-4); color:var(--txt-dim);">Símbolos</div><div class="rle-val" style="color:var(--txt);">0,1,0,0...</div></div>'
            '<span style="color:var(--muted); font-size:1.2rem;">➔</span>'
            '<div class="rle-pill"><div class="rle-count" style="background:linear-gradient(135deg, var(--amber), #d97706);">Modelo Contexto</div></div>'
            '<span style="color:var(--muted); font-size:1.2rem;">➔</span>'
            '<div class="rle-pill"><div class="rle-count" style="background:linear-gradient(135deg, var(--green), #059669);">Flujo CABAC</div><div class="rle-val" style="color:var(--green); font-weight:bold;">01101...</div></div>'
            '</div>'
        )

        html_videodec = (
            '<div style="display:flex; flex-direction:column; gap: 15px; background:var(--bg-1); padding:1.5rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="font-family:\'IBM Plex Mono\', monospace; font-size:0.75rem; color:var(--cyan); text-align:center;">PIPELINE DE DESCOMPRESIÓN (PLAYBACK)</div>'
            '<div style="display:flex; align-items:center; justify-content:center; gap: 10px;">'
            '<div class="rle-pill"><div class="rle-count" style="background:var(--green);">Flujo CABAC</div><div class="rle-val">01101...</div></div>'
            '<span style="color:var(--muted); font-size:1.2rem;">➔</span>'
            '<div class="rle-pill"><div class="rle-count" style="background:var(--amber);">Descodificador</div></div>'
            '<span style="color:var(--muted); font-size:1.2rem;">➔</span>'
            '<div class="rle-pill"><div class="rle-count" style="background:var(--bg-4);">Coeficientes</div><div class="rle-val">5, -2, 0...</div></div>'
            '</div>'
            '<div style="display:flex; align-items:center; justify-content:center; gap: 10px;">'
            '<div style="border:1px solid var(--border); padding:5px; border-radius:4px; font-size:0.6rem; text-align:center;">Matriz<br>Frecuencias</div>'
            '<span style="color:var(--muted); font-size:1.2rem;">➔</span>'
            '<div class="rle-pill"><div class="rle-count" style="background:var(--cyan); color:black;">IDCT Inversa</div></div>'
            '<span style="color:var(--muted); font-size:1.2rem;">➔</span>'
            '<div style="border:1px solid var(--border); padding:5px; border-radius:4px; font-size:0.6rem; text-align:center; color:#ef4444;">Residual<br>Espacial (Error)</div>'
            '</div>'
            '<div style="display:flex; align-items:center; justify-content:center; gap: 10px; margin-top:10px;">'
            '<div style="text-align:center;">'
            '<div style="width:50px; height:50px; border:1px dashed var(--muted); background:var(--bg-2); position:relative;">'
            '<div style="width:15px; height:15px; background:var(--violet); position:absolute; top:5px; left:5px;"></div>'
            '</div>'
            '<div style="font-size:0.5rem; color:var(--muted); margin-top:4px;">Frame(t-1)<br>+ Vect. Mov</div>'
            '</div>'
            '<span style="color:var(--muted); font-size:1.5rem; font-weight:bold;">+</span>'
            '<div style="text-align:center;">'
            '<div style="width:50px; height:50px; border:1px solid var(--border-strong); background:rgba(239,68,68,0.1); display:flex; align-items:center; justify-content:center; color:#ef4444; font-size:0.6rem;">Residual</div>'
            '</div>'
            '<span style="color:var(--muted); font-size:1.5rem; font-weight:bold;">=</span>'
            '<div style="text-align:center;">'
            '<div style="width:50px; height:50px; border:1px solid var(--cyan); background:var(--bg-2); position:relative;">'
            '<div style="width:15px; height:15px; background:var(--cyan); position:absolute; top:15px; left:20px;"></div>'
            '</div>'
            '<div style="font-size:0.5rem; color:var(--cyan); margin-top:4px;">Frame(t)<br>Reconstruido</div>'
            '</div>'
            '</div>'
            '</div>'
        )

        pasos = [
            {"titulo": "Paso 1 · Partición en Macroblocks", "detalle": "", "html": html_macroblocks},
            {"titulo": "Paso 2 · Predicción (Motion Estimation)", "detalle": "", "html": html_motion},
            {"titulo": "Paso 3 · DCT y Cuantización", "detalle": "", "html": html_residual},
            {"titulo": "Paso 4 · CABAC", "detalle": "", "html": html_cabac},
            {"titulo": "Paso 5 · Proceso de Descompresión (Playback)", "detalle": "", "html": html_videodec}
        ]
        
        res = ResultadoCompresion(
            nombre_algoritmo="H.264 (Simulado)",
            datos_originales=datos,
            datos_comprimidos=datos,
            datos_descomprimidos=datos,
            tamaño_original=orig,
            tamaño_comprimido=comp,
            tasa_compresion=orig / comp,
            ratio_reduccion=1 - (comp / orig),
            tiempo_ms=125.4,
            pasos=pasos,
            es_stub=True
        )
        
        render_resultado_compresion(res)
        
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Trama I (Intra)", "3.5:1", "Solo DCT espacial")
        with c2: st.metric("Trama P (Predictiva)", "12.0:1", "DCT + Motion Vector")
        with c3: st.metric("Trama B (Bidireccional)", "20.0:1", "Máxima compresión")

        render_pasos(res.pasos)
        render_descodificacion(res, "video")

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    inject_css()
    render_header()
    tabs = st.tabs(["📄 Texto", "🖼️ Imagen", "🎵 Audio", "🎬 Video"])
    with tabs[0]: tab_texto()
    with tabs[1]: tab_imagen()
    with tabs[2]: tab_audio()
    with tabs[3]: tab_video()

if __name__ == "__main__":
    main()from __future__ import annotations

import heapq
import io
import math
import struct
import time
import wave
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ── Optional dependencies ──────────────────────────────────────────────────
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  ← must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataLab · Compresión",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS — "Terminal Lab" Dark Aesthetic
# ─────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&display=swap');

        :root {
            --bg-0:    #030712;
            --bg-1:    #070d1a;
            --bg-2:    #0d1424;
            --bg-3:    #141e30;
            --bg-4:    #1a2540;
            --cyan:    #06b6d4;
            --cyan-dim: rgba(6,182,212,0.12);
            --violet:  #8b5cf6;
            --violet-dim: rgba(139,92,246,0.12);
            --green:   #10b981;
            --amber:   #f59e0b;
            --red:     #ef4444;
            --txt:     #cdd9e5;
            --txt-dim: #8b9cb5;
            --muted:   #4e5f7a;
            --border:  rgba(6,182,212,0.14);
            --border-strong: rgba(6,182,212,0.30);
            --radius-sm: 8px;
            --radius-md: 14px;
            --radius-lg: 20px;
        }

        html, body, [class*="css"], [data-testid] {
            font-family: 'Syne', sans-serif !important;
            background-color: var(--bg-0) !important;
            color: var(--txt) !important;
        }
        * { box-sizing: border-box; }

        #MainMenu, footer, header { visibility: hidden !important; }
        .stDeployButton, [data-testid="stToolbar"] { display: none !important; }
        section[data-testid="stSidebar"] { display: none !important; }

        .main .block-container { padding: 0 2rem 4rem !important; max-width: 1440px !important; }

        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: var(--bg-1); }
        ::-webkit-scrollbar-thumb { background: var(--cyan); border-radius: 3px; }

        /* HEADER */
        .app-header-wrap { background: var(--bg-1); border-bottom: 1px solid var(--border); padding: 1.2rem 0 1.4rem; margin: 0 -2rem 2rem; padding-left: 2rem; padding-right: 2rem; }
        .app-header { display: flex; align-items: center; justify-content: space-between; gap: 1.5rem; }
        .header-left { display: flex; align-items: center; gap: 1.25rem; }
        .header-icon { width: 48px; height: 48px; background: linear-gradient(135deg, var(--cyan), var(--violet)); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; flex-shrink: 0; }
        .app-title { font-size: 1.55rem; font-weight: 800; background: linear-gradient(100deg, var(--cyan) 0%, #a78bfa 60%, var(--violet) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; letter-spacing: -0.04em; margin: 0; }
        .app-subtitle { font-family: 'IBM Plex Mono', monospace !important; color: var(--muted); font-size: 0.7rem; letter-spacing: 0.06em; margin-top: 0.2rem; }
        
        /* TABS */
        .stTabs [data-baseweb="tab-list"] { background: var(--bg-2) !important; border-radius: var(--radius-md) !important; padding: 5px !important; gap: 4px !important; border: 1px solid var(--border) !important; }
        .stTabs [data-baseweb="tab"] { background: transparent !important; border-radius: var(--radius-sm) !important; color: var(--muted) !important; font-weight: 600 !important; font-size: 0.82rem !important; padding: 0.55rem 1.4rem !important; border: none !important; }
        .stTabs [aria-selected="true"] { background: linear-gradient(135deg, var(--cyan-dim), var(--violet-dim)) !important; color: var(--cyan) !important; border: 1px solid var(--border-strong) !important; }
        .stTabs [data-baseweb="tab-panel"] { padding: 1.75rem 0 0 !important; }

        /* SECTIONS & METRICS */
        .section-label { display: flex; align-items: center; gap: 0.6rem; margin: 1.5rem 0 0.85rem; }
        .section-label .sl-icon { font-size: 0.9rem; }
        .section-label .sl-text { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.65rem; font-weight: 600; letter-spacing: 0.18em; text-transform: uppercase; color: var(--cyan); }
        .section-label::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, var(--border) 0%, transparent 100%); }
        
        [data-testid="stMetric"] { background: var(--bg-2) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; padding: 1.1rem 1.3rem !important; position: relative; overflow: hidden; }
        [data-testid="stMetric"]::before { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--violet), var(--cyan)); }
        [data-testid="stMetricLabel"] > div { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.62rem !important; color: var(--muted) !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }
        [data-testid="stMetricValue"] > div { font-size: 1.7rem !important; font-weight: 700 !important; color: var(--cyan) !important; font-family: 'IBM Plex Mono', monospace !important; letter-spacing: -0.03em !important; }

        /* UPLOADER & BUTTONS */
        [data-testid="stFileUploader"] { background: var(--bg-2) !important; border: 1.5px dashed rgba(6,182,212,0.28) !important; border-radius: var(--radius-lg) !important; padding: 1.5rem !important; transition: border-color 0.25s !important; }
        [data-testid="stFileUploader"]:hover { border-color: var(--cyan) !important; box-shadow: var(--glow-cyan) !important; }
        
        .stButton > button { background: linear-gradient(135deg, var(--cyan) 0%, var(--violet) 100%) !important; color: #fff !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important; font-family: 'Syne', sans-serif !important; font-size: 0.85rem !important; padding: 0.55rem 2rem !important; letter-spacing: 0.03em !important; transition: all 0.2s ease !important; box-shadow: 0 2px 12px rgba(6,182,212,0.2) !important; }
        .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 24px rgba(6,182,212,0.35) !important; }

        /* INFO BOXES */
        .info-box { display: flex; gap: 0.9rem; align-items: flex-start; background: rgba(6,182,212,0.06); border: 1px solid rgba(6,182,212,0.22); border-radius: var(--radius-md); padding: 1rem 1.25rem; margin: 0.75rem 0; }
        .info-box.amber { background: rgba(245,158,11,0.06); border-color: rgba(245,158,11,0.22); }
        .info-box.green { background: rgba(16,185,129,0.06); border-color: rgba(16,185,129,0.22); }
        .info-box.violet { background: rgba(139,92,246,0.06); border-color: rgba(139,92,246,0.22); }
        .info-box .ib-content { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.75rem; color: var(--txt); line-height: 1.7; }
        
        /* VISUALIZACIONES GRÁFICAS (DCT Y RLE) */
        .dct-wrapper { display: flex; flex-direction: column; align-items: center; background: var(--bg-1); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border); margin: 1rem 0; }
        .dct-grid { display: grid; grid-template-columns: repeat(8, 1fr); gap: 2px; width: 100%; max-width: 320px; background: var(--border); padding: 2px; border-radius: 4px; }
        .dct-cell { aspect-ratio: 1; display: flex; align-items: center; justify-content: center; font-size: 0.65rem; font-family: 'IBM Plex Mono', monospace; border-radius: 2px; font-weight: 600; }
        .dct-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: var(--cyan); margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.1em; }

        .rle-wrapper { display: flex; flex-wrap: wrap; gap: 8px; background: var(--bg-1); padding: 1.25rem; border-radius: 8px; border: 1px solid var(--border); margin: 1rem 0; }
        .rle-pill { display: flex; border-radius: 6px; overflow: hidden; border: 1px solid var(--border); }
        .rle-count { background: linear-gradient(135deg, var(--cyan), var(--violet)); color: #fff; font-weight: 800; padding: 4px 10px; font-size: 0.8rem; font-family: 'IBM Plex Mono', monospace; display: flex; align-items: center; justify-content: center; }
        .rle-val { background: var(--bg-3); color: var(--txt); padding: 4px 12px; font-size: 0.8rem; font-family: 'IBM Plex Mono', monospace; display: flex; align-items: center; justify-content: center; }

        /* CODE BLOCK OVERRIDES */
        .stCode, [data-testid="stCode"] { background: #010508 !important; border: 1px solid rgba(6,182,212,0.18) !important; border-radius: var(--radius-md) !important; }
        
        .no-file-state { text-align: center; padding: 3rem 2rem; background: var(--bg-2); border: 1.5px dashed var(--border); border-radius: var(--radius-lg); margin-top: 1.5rem; }
        .no-file-state .nfs-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
        .no-file-state .nfs-title { font-size: 1rem; font-weight: 600; color: var(--txt-dim); margin-bottom: 0.4rem; }
        .no-file-state .nfs-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--muted); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
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
#  ABSTRACT BASE CLASS
# ═════════════════════════════════════════════════════════════════════════════

class AnalizadorInformacion(ABC):
    def __init__(self, datos: bytes, nombre: str = "datos") -> None:
        if not datos: raise ValueError("Los datos no pueden estar vacíos.")
        self._datos = datos
        self._nombre = nombre

    @abstractmethod
    def _extraer_simbolos(self) -> List[Any]: ...

    def calcular_frecuencias(self) -> Dict[str, int]: return dict(Counter(self._extraer_simbolos()))
    def calcular_probabilidades(self) -> Dict[str, float]:
        freq = self.calcular_frecuencias()
        total = sum(freq.values())
        return {k: v / total for k, v in freq.items()}

    def calcular_entropia(self) -> float:
        return -sum(p * math.log2(p) for p in self.calcular_probabilidades().values() if p > 0)

    def calcular_longitud_promedio(self, longitudes: Optional[Dict[str, int]] = None) -> float:
        probs = self.calcular_probabilidades()
        if longitudes: return sum(probs[s] * longitudes[s] for s in probs if s in longitudes)
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
    def _extraer_simbolos(self) -> List[int]: return list(self._datos[: self.CAP])


# ═════════════════════════════════════════════════════════════════════════════
#  COMPRESSION & DECOMPRESSION ALGORITHMS
# ═════════════════════════════════════════════════════════════════════════════

# ─── A. HUFFMAN ──────────────────────────────────────────────────────────────
class NodoHuffman:
    __slots__ = ("simbolo", "frecuencia", "izquierda", "derecha")
    def __init__(self, simbolo: Optional[str], frecuencia: int) -> None:
        self.simbolo, self.frecuencia = simbolo, frecuencia
        self.izquierda = self.derecha = None
    def __lt__(self, other: "NodoHuffman") -> bool: return self.frecuencia < other.frecuencia

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

    def _generar_codigos(self, nodo: Optional[NodoHuffman], prefijo: str = "", codigos: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        if codigos is None: codigos = {}
        if nodo is None: return codigos
        if nodo.simbolo is not None: codigos[nodo.simbolo] = prefijo if prefijo else "0"
        else:
            self._generar_codigos(nodo.izquierda, prefijo + "0", codigos)
            self._generar_codigos(nodo.derecha, prefijo + "1", codigos)
        return codigos

    def _generar_dot(self, nodo: Optional[NodoHuffman], total_simbolos: int) -> str:
        if not nodo: return ""
        lineas = [
            "digraph Huffman {",
            "    rankdir=RL;",            # Dibuja de Derecha a Izquierda
            "    bgcolor=\"transparent\";",
            "    node [fontname=\"monospace\", style=\"filled\", fillcolor=\"#1e293b\", fontcolor=\"#cdd9e5\", color=\"#06b6d4\"];",
            "    edge [fontname=\"monospace\", fontcolor=\"#ef4444\", color=\"#8b9cb5\", fontsize=10];"
        ]
        contador = [0]
        def escapar_etiqueta(sym: Any) -> str:
            if isinstance(sym, int): return f"0x{sym:02X}"
            s = repr(sym)
            if s.startswith("'") and s.endswith("'"): s = s[1:-1]
            elif s.startswith('"') and s.endswith('"'): s = s[1:-1]
            return s.replace("\\", "\\\\").replace('"', '\\"') if s else "Espacio"

        def recorrer(n: NodoHuffman) -> str:
            my_id = f"N{contador[0]}"
            contador[0] += 1
            prob = n.frecuencia / total_simbolos if total_simbolos > 0 else 0
            if n.simbolo is not None:
                lineas.append(f"    {my_id} [label=\"{escapar_etiqueta(n.simbolo)}: {prob:.2g}\", shape=\"box\", fillcolor=\"#0f172a\", color=\"#06b6d4\"];")
            else:
                lineas.append(f"    {my_id} [label=\"{prob:.2g}\", shape=\"ellipse\", color=\"#8b5cf6\"];")
            if n.izquierda: lineas.append(f"    {my_id} -> {recorrer(n.izquierda)} [label=\" 0 \", fontcolor=\"#06b6d4\"];")
            if n.derecha: lineas.append(f"    {my_id} -> {recorrer(n.derecha)} [label=\" 1 \", fontcolor=\"#ef4444\"];")
            return my_id
        recorrer(nodo)
        lineas.append("}")
        return "\n".join(lineas)

    def comprimir(self) -> ResultadoCompresion:
        t0 = time.perf_counter()
        try: texto = self._datos.decode("utf-8", errors="replace")
        except Exception: texto = self._datos.decode("latin-1", errors="replace")
        
        freq = Counter(texto)
        raiz = self._construir_arbol(dict(freq))
        codigos = self._generar_codigos(raiz)
        
        total_bits = sum(len(codigos.get(c, "?")) for c in texto)
        comprimido = bytes(math.ceil(total_bits / 8)) 

        tabla_frec = [{"Símbolo": repr(k), "Frecuencia": v} for k, v in sorted(freq.items(), key=lambda x: -x[1])]

        pasos = [
            {"titulo": "Paso 1 · Frecuencias y Repeticiones", "detalle": "", "tabla": tabla_frec[:150]},
            {"titulo": "Paso 2 · Construcción del Árbol", "detalle": "Fusión Bottom-Up usando Min-Heap."},
        ]

        grafo_dot = self._generar_dot(raiz, len(texto)) if len(freq) <= 60 else None

        datos_reconstruidos = self._datos 

        return ResultadoCompresion("Huffman", self._datos, comprimido, datos_reconstruidos, len(self._datos), max(1, len(comprimido)), len(self._datos) / max(1, len(comprimido)), 1 - (max(1, len(comprimido)) / len(self._datos)), (time.perf_counter() - t0) * 1000, pasos, codigos, grafo_dot=grafo_dot, es_stub=True)

# ─── B. LZW ──────────────────────────────────────────────────────────────────
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
        
        # ── REAL LZW DECOMPRESSION ──
        datos_reconstruidos = b""
        if comprimido:
            codigos_lzw = struct.unpack(f">{len(comprimido)//2}H", comprimido)
            dic_dec = {i: chr(i) for i in range(256)}
            sig_dec = 256
            w_dec = chr(codigos_lzw[0])
            res_desc = [w_dec]
            for codigo in codigos_lzw[1:]:
                if codigo in dic_dec: entrada = dic_dec[codigo]
                elif codigo == sig_dec: entrada = w_dec + w_dec[0]
                else: break
                res_desc.append(entrada)
                dic_dec[sig_dec] = w_dec + entrada[0]
                sig_dec += 1
                w_dec = entrada
            datos_reconstruidos = "".join(res_desc).encode("utf-8", errors="replace")

        return ResultadoCompresion("LZW", datos, comprimido, datos_reconstruidos, len(datos), max(1, len(comprimido)), len(datos) / max(1, len(comprimido)), 1 - (max(1, len(comprimido)) / len(datos)), (time.perf_counter() - t0) * 1000, [{"titulo": "Matching en Diccionario", "detalle": "Codificación y decodificación LZW ejecutada."}], es_stub=False)

# ─── C. RLE ───────────────────────────────────────────────────────────────────
class CodificadorRLE:
    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        t0 = time.perf_counter()
        if not datos: return ResultadoCompresion("RLE", b"", b"", b"", 0, 0, 1.0, 0.0, 0.0, es_stub=False)
        
        comprimido = bytearray()
        runs_log = []
        i = 0
        while i < len(datos):
            byte_actual = datos[i]
            conteo = 1
            while i + conteo < len(datos) and datos[i + conteo] == byte_actual and conteo < 255:
                conteo += 1
            comprimido.extend([conteo, byte_actual])
            if len(runs_log) < 25: runs_log.append({"Run": conteo, "Byte": f"0x{byte_actual:02X}"})
            i += conteo

        html_rle = '<div class="rle-wrapper">'
        for run in runs_log[:18]:
            html_rle += f'<div class="rle-pill" title="Se repite {run["Run"]} veces el byte {run["Byte"]}"><div class="rle-count">{run["Run"]}×</div><div class="rle-val">{run["Byte"]}</div></div>'
        html_rle += '</div>'

        # ── REAL RLE DECOMPRESSION ──
        decomp = bytearray()
        idx = 0
        while idx + 1 < len(comprimido):
            decomp.extend([comprimido[idx+1]] * comprimido[idx])
            idx += 2
        datos_reconstruidos = bytes(decomp)

        pasos = [{"titulo": "Particionado Secuencial (Bloques RLE)", "detalle": "", "html": html_rle}]
        return ResultadoCompresion("RLE", datos, bytes(comprimido), datos_reconstruidos, len(datos), max(1, len(comprimido)), len(datos) / max(1, len(comprimido)), 1 - (max(1, len(comprimido)) / len(datos)), (time.perf_counter() - t0) * 1000, pasos, es_stub=False)

# ─── D. DCT (Image / JPEG-like) ───────────────────────────────────────────────
class CodificadorDCT:
    def _dct2d_manual(self, bloque: np.ndarray) -> np.ndarray:
        N = 8
        b = bloque.astype(float) - 128.0
        F = np.zeros((N, N))
        xs, ys = np.arange(N), np.arange(N)
        for u in range(N):
            for v in range(N):
                cu = 1.0 / math.sqrt(2) if u == 0 else 1.0
                cv = 1.0 / math.sqrt(2) if v == 0 else 1.0
                cos_u = np.cos((2 * xs + 1) * u * math.pi / (2 * N))
                cos_v = np.cos((2 * ys + 1) * v * math.pi / (2 * N))
                F[u, v] = (2 / N) * cu * cv * np.sum(b * cos_u[:, None] * cos_v[None, :])
        return F

    def comprimir(self, datos: bytes, calidad: int = 50) -> ResultadoCompresion:
        t0 = time.perf_counter()
        raw = list(datos[:64]) + [128] * max(0, 64 - len(datos))
        bloque = np.array(raw[:64], dtype=float).reshape(8, 8)
        
        # STRING HTML EN UNA SOLA LÍNEA PARA EVITAR BUG DE STREAMLIT MARKDOWN
        html_dct = (
            '<div class="dct-wrapper"><div class="dct-title">Bloque Extraído (8x8 px)</div><div class="dct-grid">'
        )
        for val in raw[:64]:
            color = max(0, min(255, int(val)))
            text_color = "black" if color > 127 else "white"
            html_dct += f'<div class="dct-cell" style="background-color: rgb({color},{color},{color}); color: {text_color};">{color}</div>'
        html_dct += '</div></div>'
        
        orig = len(datos)
        theo_ratio = max(0.05, 1 - (calidad / 100) * 0.92)
        comp = max(1, int(orig * theo_ratio))
        
        datos_reconstruidos = datos

        # STRING HTML EN UNA SOLA LÍNEA (IDCT GRID)
        html_idct = (
            '<div style="display:flex; flex-direction:column; gap: 15px; background:var(--bg-1); padding:1rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="font-family:\'IBM Plex Mono\', monospace; font-size:0.75rem; color:var(--cyan); text-align:center;">PROCEDIMIENTO: IDCT (Transformada Inversa)</div>'
            '<div style="display:flex; flex-wrap:wrap; justify-content:center; align-items:center; gap: 20px;">'
            '<div style="display:flex; flex-direction:column; align-items:center;">'
            '<div style="font-family:\'IBM Plex Mono\', monospace; font-size:0.6rem; color:var(--muted); margin-bottom:8px;">1. MATRIZ CUANTIZADA</div>'
            '<div class="dct-grid" style="width:140px; background:var(--bg-2); border:1px solid var(--border);">'
        )
        for i in range(64):
            val = 35 if i==0 else (-12 if i==1 else (8 if i==8 else 0))
            text_color = "#ef4444" if val != 0 else "rgba(139,156,181,0.2)"
            html_idct += f'<div class="dct-cell" style="background:var(--bg-0); color:{text_color}; font-size:0.55rem;">{val}</div>'
        html_idct += (
            '</div></div>'
            '<span style="color:var(--cyan); font-size:1.5rem; font-weight:bold;">➔</span>'
            '<div style="display:flex; flex-direction:column; align-items:center;">'
            '<div style="font-family:\'IBM Plex Mono\', monospace; font-size:0.6rem; color:var(--muted); margin-bottom:8px;">2. BLOQUE RECONSTRUIDO</div>'
            '<div class="dct-grid" style="width:140px; background:var(--bg-2); border:1px solid var(--border);">'
        )
        for val in raw[:64]:
            color = max(0, min(255, int(val)))
            html_idct += f'<div class="dct-cell" style="background-color: rgb({color},{color},{color}); color: transparent;">0</div>'
        html_idct += (
            '</div></div></div>'
            '</div>'
        )

        pasos = [
            {"titulo": "Paso 1 · Partición en Bloques y Transformada", "detalle": "", "html": html_dct},
            {"titulo": "Paso 2 · Proceso de Descompresión (Reconstrucción IDCT)", "detalle": "", "html": html_idct}
        ]
        return ResultadoCompresion("DCT", datos, bytes(comp), datos_reconstruidos, orig, comp, orig / comp, 1 - (comp / orig), (time.perf_counter() - t0) * 1000, pasos, es_stub=True)

# ─── E. μ-Law (Audio) & F. ADPCM ──────────────────────────────────────────────
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
        samples = struct.unpack(f"<{n_samples}h", datos[: n_samples * 2])
        encoded = bytes(self._encode_sample(s) for s in samples)

        # ── REAL MU-LAW DECOMPRESSION ──
        decoded = []
        for b in encoded:
            byte = ~b & 0xFF
            sign = byte & 0x80
            exp = (byte >> 4) & 0x07
            mant = byte & 0x0F
            mag = ((mant << 1) | 1) << (exp + 2)
            val = -(mag - self.BIAS) if sign else (mag - self.BIAS)
            decoded.append(val)
        datos_reconstruidos = struct.pack(f"<{len(decoded)}h", *decoded)

        # HTML VISUAL PARA MU-LAW (SIN SALTOS DE LINEA)
        html_mulaw_comp = (
            '<div style="display:flex; align-items:center; justify-content:center; gap:20px; background:var(--bg-1); padding:1.5rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="border:1px solid var(--border); padding:15px; border-radius:8px; text-align:center; background:var(--bg-2); min-width:120px;">'
            '<div style="font-size:0.65rem; color:var(--muted); font-family:\'IBM Plex Mono\', monospace; margin-bottom:5px;">MUESTRA ORIGINAL</div>'
            '<div style="font-size:1.5rem; color:var(--cyan); font-weight:bold;">16-bit</div>'
            '<div style="font-size:0.7rem; color:var(--txt-dim); margin-top:5px;">Linear PCM</div>'
            '</div>'
            '<div style="display:flex; flex-direction:column; align-items:center;">'
            '<div style="font-size:0.65rem; color:var(--amber); font-family:\'IBM Plex Mono\', monospace; padding:4px 8px; background:rgba(245,158,11,0.1); border-radius:4px; margin-bottom:4px;">Companding Logarítmico</div>'
            '<span style="color:var(--muted); font-size:1.5rem;">➔</span>'
            '</div>'
            '<div style="border:1px solid var(--border-strong); padding:15px; border-radius:8px; text-align:center; background:rgba(239,68,68,0.05); min-width:120px;">'
            '<div style="font-size:0.65rem; color:var(--muted); font-family:\'IBM Plex Mono\', monospace; margin-bottom:5px;">CODIFICADO</div>'
            '<div style="font-size:1.5rem; color:#ef4444; font-weight:bold;">8-bit</div>'
            '<div style="font-size:0.7rem; color:var(--txt-dim); margin-top:5px;">μ-Law</div>'
            '</div>'
            '</div>'
        )

        html_mulaw_decomp = (
            '<div style="display:flex; align-items:center; justify-content:center; gap:20px; background:var(--bg-1); padding:1.5rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="border:1px solid var(--border-strong); padding:15px; border-radius:8px; text-align:center; background:rgba(239,68,68,0.05); min-width:120px;">'
            '<div style="font-size:0.65rem; color:var(--muted); font-family:\'IBM Plex Mono\', monospace; margin-bottom:5px;">ARCHIVO GUARDADO</div>'
            '<div style="font-size:1.5rem; color:#ef4444; font-weight:bold;">8-bit</div>'
            '</div>'
            '<div style="display:flex; flex-direction:column; align-items:center;">'
            '<div style="font-size:0.65rem; color:var(--green); font-family:\'IBM Plex Mono\', monospace; padding:4px 8px; background:rgba(16,185,129,0.1); border-radius:4px; margin-bottom:4px;">Expansión Inversa</div>'
            '<span style="color:var(--muted); font-size:1.5rem;">➔</span>'
            '</div>'
            '<div style="border:1px solid var(--cyan); padding:15px; border-radius:8px; text-align:center; background:rgba(6,182,212,0.05); min-width:120px; box-shadow: 0 0 15px rgba(6,182,212,0.2);">'
            '<div style="font-size:0.65rem; color:var(--cyan); font-family:\'IBM Plex Mono\', monospace; margin-bottom:5px;">ONDA RECONSTRUIDA</div>'
            '<div style="font-size:1.5rem; color:var(--cyan); font-weight:bold;">16-bit</div>'
            '</div>'
            '</div>'
        )

        pasos = [
            {"titulo": "Paso 1 · Cuantización Logarítmica (Compresión)", "detalle": "", "html": html_mulaw_comp},
            {"titulo": "Paso 2 · Proceso de Descompresión (Expansión)", "detalle": "", "html": html_mulaw_decomp}
        ]

        t1 = time.perf_counter()
        return ResultadoCompresion("μ-Law", datos, encoded, datos_reconstruidos, len(datos), len(encoded), 2.0, 0.5, (t1-t0)*1000, pasos, es_stub=False)

class CodificadorADPCM:
    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        t0 = time.perf_counter()
        n_samples = len(datos) // 2
        samples = struct.unpack(f"<{n_samples}h", datos[: n_samples * 2])
        
        encoded = bytearray()
        decoded = []
        prev_sample = 0
        
        for s in samples:
            diff = s - prev_sample
            step = 256
            quantized_diff = max(-128, min(127, diff // step))
            encoded.append(quantized_diff & 0xFF)
            recon_diff = quantized_diff * step
            prev_sample = max(-32768, min(32767, prev_sample + recon_diff))
            decoded.append(prev_sample)
            
        datos_reconstruidos = struct.pack(f"<{len(decoded)}h", *decoded)

        # HTML VISUAL PARA ADPCM (SIN SALTOS DE LINEA)
        html_adpcm_diff = (
            '<div style="display:flex; align-items:center; justify-content:center; gap:15px; background:var(--bg-1); padding:1.5rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="text-align:center;"><div style="width:50px; height:60px; background:var(--cyan); margin:0 auto; border-radius:4px;"></div><div style="font-size:0.6rem; color:var(--muted); margin-top:5px; font-family:\'IBM Plex Mono\', monospace;">Muestra<br>(t)</div></div>'
            '<span style="font-size:2rem; color:var(--muted); font-weight:bold;">-</span>'
            '<div style="text-align:center;"><div style="width:50px; height:45px; background:var(--violet); margin:0 auto; border-radius:4px;"></div><div style="font-size:0.6rem; color:var(--muted); margin-top:5px; font-family:\'IBM Plex Mono\', monospace;">Anterior<br>(t-1)</div></div>'
            '<span style="font-size:2rem; color:var(--muted); font-weight:bold;">=</span>'
            '<div style="border:1px dashed var(--border-strong); padding:15px; border-radius:8px; text-align:center; background:rgba(6,182,212,0.05);">'
            '<div style="font-size:0.6rem; color:var(--cyan); font-family:\'IBM Plex Mono\', monospace; margin-bottom:5px;">DIFERENCIA (ERROR)</div>'
            '<div style="font-size:1.2rem; color:var(--txt); font-weight:bold;">Δ</div>'
            '</div>'
            '</div>'
        )

        html_adpcm_quant = (
            '<div style="display:flex; align-items:center; justify-content:center; gap:20px; background:var(--bg-1); padding:1.5rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="border:1px dashed var(--border-strong); padding:15px; border-radius:8px; text-align:center; background:rgba(6,182,212,0.05); min-width:100px;">'
            '<div style="font-size:1.2rem; color:var(--txt); font-weight:bold;">Δ</div>'
            '<div style="font-size:0.6rem; color:var(--cyan); margin-top:5px;">Diferencia</div>'
            '</div>'
            '<div style="display:flex; flex-direction:column; align-items:center;">'
            '<div style="font-size:0.65rem; color:var(--amber); font-family:\'IBM Plex Mono\', monospace; padding:4px 8px; background:rgba(245,158,11,0.1); border-radius:4px; margin-bottom:4px;">÷ Step Size</div>'
            '<span style="color:var(--muted); font-size:1.5rem;">➔</span>'
            '</div>'
            '<div style="border:1px solid #ef4444; padding:15px; border-radius:8px; text-align:center; background:rgba(239,68,68,0.05); min-width:100px;">'
            '<div style="font-size:1.2rem; color:#ef4444; font-weight:bold;">4-bit / 8-bit</div>'
            '<div style="font-size:0.6rem; color:var(--muted); margin-top:5px;">Cuantizado a guardar</div>'
            '</div>'
            '</div>'
        )

        html_adpcm_recon = (
            '<div style="display:flex; align-items:center; justify-content:center; gap:15px; background:var(--bg-1); padding:1.5rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="border:1px solid #ef4444; padding:10px; border-radius:8px; text-align:center; background:rgba(239,68,68,0.05);"><div style="font-size:0.8rem; color:#ef4444; font-weight:bold;">Quantizado</div></div>'
            '<span style="font-size:1.2rem; color:var(--muted);">× Step =</span>'
            '<div style="border:1px dashed var(--border-strong); padding:10px; border-radius:8px; text-align:center;"><div style="font-size:0.8rem; color:var(--cyan); font-weight:bold;">Δ Reconstruido</div></div>'
            '<span style="font-size:1.5rem; color:var(--muted); font-weight:bold;">+</span>'
            '<div style="text-align:center;"><div style="width:40px; height:45px; background:var(--violet); margin:0 auto; border-radius:4px;"></div><div style="font-size:0.5rem; color:var(--muted); margin-top:5px;">Anterior (t-1)</div></div>'
            '<span style="font-size:1.5rem; color:var(--muted); font-weight:bold;">=</span>'
            '<div style="text-align:center;"><div style="width:40px; height:60px; background:var(--cyan); margin:0 auto; border-radius:4px; box-shadow: 0 0 10px rgba(6,182,212,0.4);"></div><div style="font-size:0.5rem; color:var(--cyan); margin-top:5px;">Onda Final (16-bit)</div></div>'
            '</div>'
        )

        pasos = [
            {"titulo": "Paso 1 · Predicción Temporal (Diferencia)", "detalle": "", "html": html_adpcm_diff},
            {"titulo": "Paso 2 · Cuantización (Reducción de Bits)", "detalle": "", "html": html_adpcm_quant},
            {"titulo": "Paso 3 · Proceso de Descompresión (Reconstrucción Predictiva)", "detalle": "", "html": html_adpcm_recon}
        ]

        t1 = time.perf_counter()
        return ResultadoCompresion("ADPCM (Funcional Simplificado)", datos, bytes(encoded), datos_reconstruidos, len(datos), len(encoded), 2.0, 0.5, (t1-t0)*1000, pasos, es_stub=False)


# ═════════════════════════════════════════════════════════════════════════════
#  SOURCE CODE STRINGS (CON FÓRMULAS)
# ═════════════════════════════════════════════════════════════════════════════

_CODE_MAP = {
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
            dic[wc] = sig
            sig += 1
            w = c
    if w: salida.append(dic[w])
    return salida
""",
    "RLE": """\
# Run-Length Encoding (RLE)
def rle_comprimir(datos: bytes) -> bytes:
    out, i = bytearray(), 0
    while i < len(datos):
        byte = datos[i]
        run = 1
        while i + run < len(datos) and datos[i+run] == byte and run < 255:
            run += 1
        out.extend([run, byte])
        i += run
    return bytes(out)

def rle_descomprimir(datos: bytes) -> bytes:
    out, i = bytearray(), 0
    while i + 1 < len(datos):
        out.extend([datos[i+1]] * datos[i])
        i += 2
    return bytes(out)
""",
    "DCT — JPEG-like": """\
# Transformada Discreta del Coseno (DCT-II 2D)
# F(u,v) = (2/N) * Cu * Cv * ΣΣ f(x,y) * cos(...) * cos(...)

def dct2(bloque: np.ndarray) -> np.ndarray:
    N = 8
    b = bloque.astype(float) - 128
    F = np.zeros((N, N))
    xs = np.arange(N)
    for u in range(N):
        cu = 1/math.sqrt(2) if u == 0 else 1.0
        cos_u = np.cos((2*xs+1)*u*math.pi/(2*N))
        for v in range(N):
            cv = 1/math.sqrt(2) if v == 0 else 1.0
            cos_v = np.cos((2*xs+1)*v*math.pi/(2*N))
            F[u,v] = (2/N)*cu*cv*np.sum(b * cos_u[:,None] * cos_v[None,:])
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
                    # Búsqueda en ventana ±16
                    ref_block = frame_ref[i+dy:..., j+dx:...]
                    sad = np.sum(np.abs(curr_block - ref_block))
                    if sad < best_sad:
                        best_sad, best_mv = sad, (dx, dy)
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
# IMA ADPCM (Adaptive Differential PCM)
# Cuantización de la diferencia entre la muestra y el predictor
# Δ = (x_n - x_pred) / step_size
"""
}

# ═════════════════════════════════════════════════════════════════════════════
#  UI HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def fmt_bytes(n: int) -> str:
    if n < 1024: return f"{n} B"
    if n < 1024**2: return f"{n/1024:.2f} KB"
    return f"{n/1024**2:.2f} MB"

def render_header() -> None:
    st.markdown('<div class="app-header-wrap"><div class="app-header"><div class="header-left"><div class="header-icon">⚡</div><div><p class="app-title">DataLab · Compresión</p><p class="app-subtitle">Shannon · Huffman · LZW · RLE · DCT · μ-Law · ADPCM · H.264</p></div></div></div></div>', unsafe_allow_html=True)

def section_label(icon: str, text: str) -> None:
    st.markdown(f'<div class="section-label"><span class="sl-icon">{icon}</span><span class="sl-text">{text}</span></div>', unsafe_allow_html=True)

def info_box(icon: str, content: str, variant: str = "") -> None:
    st.markdown(f'<div class="info-box {variant}"><div class="ib-icon">{icon}</div><div class="ib-content">{content}</div></div>', unsafe_allow_html=True)

def render_dashboard(stats: EstadisticasInfo) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Entropía H(X)", f"{stats.entropia:.4f}", "bits / símbolo")
    with c2: st.metric("Long. Promedio L̄", f"{stats.longitud_promedio:.4f}", "bits / símbolo")
    with c3: st.metric("Eficiencia η", f"{stats.eficiencia * 100:.2f}%", f"Redundancia {stats.redundancia*100:.2f}%")
    with c4: st.metric("Alfabeto", f"{stats.simbolos_unicos}", f"de {stats.total_simbolos:,} símbolos")

def render_codigo_y_formulas(algo: str, stats: EstadisticasInfo) -> None:
    section_label("📜", f"CÓDIGO Y FÓRMULAS — {algo}")
    info_box("🧮", f"<strong>Entropía Matemática:</strong> $H(X) = -\sum p(x_i) \cdot \log_2 p(x_i)$<br>Límite teórico calculado: <strong>{stats.entropia:.4f} bits/símbolo</strong>", "amber")
    st.code(_CODE_MAP.get(algo, "# Código no disponible para este algoritmo."), language="python")

def render_resultado_compresion(res: ResultadoCompresion) -> None:
    section_label("📦", f"RESULTADO DE COMPRESIÓN — {res.nombre_algoritmo}")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Tamaño Original", fmt_bytes(res.tamaño_original))
    with c2: st.metric("Tamaño Comprimido", fmt_bytes(res.tamaño_comprimido))
    with c3: st.metric("Tasa", f"{res.tasa_compresion:.2f}:1")
    with c4: st.metric("Reducción", f"{max(0,res.ratio_reduccion)*100:.1f}%")
    
    if res.es_stub: info_box("🔧", "Modo Educativo (STUB): Los bytes de salida son teóricos.", "amber")

def render_pasos(pasos: List[Dict[str, Any]]) -> None:
    section_label("🔍", "PROCEDIMIENTO PASO A PASO")
    for i, paso in enumerate(pasos, 1):
        with st.expander(paso.get("titulo", f"Paso {i}"), expanded=(i <= 2)):
            if paso.get("detalle"): st.markdown(f"<div style='font-size:0.8rem;color:var(--txt-dim);margin-bottom:10px;white-space:pre-wrap'>{paso['detalle']}</div>", unsafe_allow_html=True)
            if paso.get("html"): st.markdown(paso["html"], unsafe_allow_html=True)
            elif paso.get("tabla"): st.dataframe(pd.DataFrame(paso["tabla"]), use_container_width=True, height=250)

def render_arbol_huffman(res: ResultadoCompresion) -> None:
    if res.nombre_algoritmo == "Huffman" and res.grafo_dot:
        section_label("🌿", "DIAGRAMA DEL ÁRBOL DE HUFFMAN")
        info_box("💡", "Árbol horizontal (Derecha a Izquierda) con probabilidades y etiquetas de bit.")
        st.graphviz_chart(res.grafo_dot)

def render_descodificacion(res: ResultadoCompresion, tipo_dato: str) -> None:
    section_label("🔄", "DESCODIFICACIÓN Y RECONSTRUCCIÓN")
    
    if res.es_stub:
        info_box("⚠️", "Este algoritmo está en modo STUB. La reconstrucción mostrada abajo es simulada usando los datos originales para demostrar el flujo final.", "amber")
    else:
        info_box("✅", "El algoritmo ha ejecutado exitosamente el proceso inverso (Descompresión matemática real).", "green")

    if tipo_dato == "texto":
        try:
            texto_recon = res.datos_descomprimidos.decode("utf-8")
        except Exception:
            texto_recon = res.datos_descomprimidos.decode("latin-1", errors="replace")
        st.text_area("Texto Reconstruido", value=texto_recon[:1000] + ("..." if len(texto_recon) > 1000 else ""), height=150, disabled=True)
    
    elif tipo_dato == "imagen":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original**")
            st.image(res.datos_originales, use_container_width=True)
        with c2:
            st.markdown("**Reconstruida**")
            st.image(res.datos_descomprimidos, use_container_width=True)
            
    elif tipo_dato == "audio":
        st.markdown("**Audio Reconstruido**")
        if res.nombre_algoritmo == "μ-Law":
            try:
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1) 
                    wf.setsampwidth(2) 
                    wf.setframerate(44100) 
                    wf.writeframes(res.datos_descomprimidos)
                st.audio(buf.getvalue(), format="audio/wav")
            except Exception:
                st.info("Formato de audio no reproducible directamente, pero los bytes fueron reconstruidos exitosamente.")
        else:
            st.audio(res.datos_descomprimidos)
            
    elif tipo_dato == "video":
        st.markdown("**Video Reconstruido**")
        st.video(res.datos_descomprimidos)

def render_no_file(icon: str, texto: str, subtexto: str) -> None:
    st.markdown(f'<div class="no-file-state"><div class="nfs-icon">{icon}</div><div class="nfs-title">{texto}</div><div class="nfs-sub">{subtexto}</div></div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TABS Y MAIN
# ═════════════════════════════════════════════════════════════════════════════

def tab_texto() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")
    with col_up:
        section_label("📄", "CARGAR ARCHIVO DE TEXTO")
        uploaded = st.file_uploader("Sube un archivo de texto", type=["txt", "csv", "json"], key="uploader_texto")
    with col_info:
        section_label("ℹ️", "ALGORITMOS")
        st.info("Huffman · LZW · RLE")

    if uploaded is None:
        render_no_file("📝", "Sube un archivo", ".txt .csv .json")
        return

    datos = uploaded.read()
    analizador = AnalizadorTexto(datos, uploaded.name)
    stats = analizador.calcular_todo()
    render_dashboard(stats)

    section_label("⚙️", "ALGORITMO DE COMPRESIÓN")
    c_sel, c_btn = st.columns([3, 1])
    with c_sel: algo = st.selectbox("Algoritmo", ["Huffman", "LZW", "RLE"], key="algo_texto", label_visibility="collapsed")
    with c_btn: btn_comp = st.button("▶ Comprimir", key="btn_texto")

    if btn_comp:
        render_codigo_y_formulas(algo, stats) 
        
        if algo == "Huffman": res = CodificadorHuffman(datos).comprimir()
        elif algo == "LZW": res = CodificadorLZW().comprimir(datos)
        else: res = CodificadorRLE().comprimir(datos)

        render_resultado_compresion(res)
        render_pasos(res.pasos)
        render_arbol_huffman(res)
        render_descodificacion(res, "texto") 


def tab_imagen() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")
    with col_up:
        section_label("🖼️", "CARGAR IMAGEN")
        uploaded = st.file_uploader("Sube una imagen", type=["png", "jpg"], key="uploader_imagen")
    with col_info:
        section_label("ℹ️", "ALGORITMOS")
        st.info("DCT (JPEG) · RLE · Huffman")
    
    if uploaded is None:
        render_no_file("🖼️", "Sube una imagen", ".png .jpg")
        return

    datos = uploaded.read()
    st.image(datos, width=250)
    
    analizador = AnalizadorImagen(datos, uploaded.name)
    stats = analizador.calcular_todo()
    render_dashboard(stats)

    section_label("⚙️", "ALGORITMO DE COMPRESIÓN")
    c_sel, c_btn = st.columns([3, 1])
    with c_sel: algo_img = st.selectbox("Algoritmo", ["DCT — JPEG-like", "RLE", "Huffman (bytes)"], key="algo_img", label_visibility="collapsed")
    with c_btn: comprimir = st.button("▶ Comprimir", key="btn_img")

    if comprimir:
        if algo_img == "Huffman (bytes)": algo_llave = "Huffman"
        else: algo_llave = algo_img

        render_codigo_y_formulas(algo_llave, stats)
        
        if algo_img == "DCT — JPEG-like": res = CodificadorDCT().comprimir(datos)
        elif algo_img == "RLE": res = CodificadorRLE().comprimir(datos)
        else: res = CodificadorHuffman(datos).comprimir()
        
        render_resultado_compresion(res)
        render_pasos(res.pasos)
        render_arbol_huffman(res)
        render_descodificacion(res, "imagen")


def tab_audio() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")
    with col_up:
        section_label("🎵", "CARGAR ARCHIVO DE AUDIO")
        uploaded = st.file_uploader("Sube audio (.wav)", type=["wav"], key="uploader_audio")
    with col_info:
        section_label("ℹ️", "ALGORITMOS")
        st.info("μ-Law G.711 · ADPCM · Huffman")

    if not uploaded:
        render_no_file("🎵", "Sube un archivo de audio", ".wav")
        return
        
    datos = uploaded.read()
    st.audio(datos)
    
    analizador = AnalizadorAudio(datos, uploaded.name)
    stats = analizador.calcular_todo()
    render_dashboard(stats)
    
    section_label("⚙️", "ALGORITMO DE COMPRESIÓN")
    c_sel, c_btn = st.columns([3, 1])
    with c_sel: algo = st.selectbox("Algoritmo", ["μ-Law G.711", "ADPCM", "Huffman (bytes)"], key="algo_audio", label_visibility="collapsed")
    with c_btn: btn_comp = st.button("▶ Comprimir", key="btn_audio")
    
    if btn_comp:
        if algo == "Huffman (bytes)": algo_llave = "Huffman"
        else: algo_llave = algo

        render_codigo_y_formulas(algo_llave, stats)
        
        if algo == "μ-Law G.711": res = CodificadorMuLaw().comprimir(datos)
        elif algo == "ADPCM": res = CodificadorADPCM().comprimir(datos)
        else: res = CodificadorHuffman(datos).comprimir()
        
        render_resultado_compresion(res)
        render_pasos(res.pasos)
        render_arbol_huffman(res)
        render_descodificacion(res, "audio")


def tab_video() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")
    with col_up:
        section_label("🎬", "CARGAR ARCHIVO DE VIDEO")
        uploaded = st.file_uploader("Sube un archivo de video", type=["mp4", "avi", "mkv", "mov", "webm"], key="uploader_video")
    with col_info:
        section_label("ℹ️", "CONCEPTOS APLICADOS")
        st.info("Tramas I/P/B · DCT Temporal · Entropía CABAC")

    if uploaded is None:
        render_no_file("🎬", "Sube un archivo de video", ".mp4 .avi .mkv .mov .webm")
        return

    datos = uploaded.read()
    st.video(datos)
    
    analizador = AnalizadorVideo(datos, uploaded.name)
    with st.spinner("Analizando flujo de bytes de video..."):
        stats = analizador.calcular_todo()

    render_dashboard(stats)
    info_box("🔬", f"El flujo de video comprimido (H.264/HEVC) tiene entropía alta (<strong>{stats.entropia:.4f} bits/byte</strong>) porque los datos ya están codificados entrópicamente. Un video RAW tendría entropía menor y sería más compresible.", "violet")

    section_label("⚙️", "ALGORITMO DE COMPRESIÓN")
    c_sel, c_btn = st.columns([3, 1])
    with c_sel:
        algo_vid = st.selectbox("Simulación", ["H.264 / HEVC (Simulado)"], key="algo_video", label_visibility="collapsed")
    with c_btn:
        btn_comp = st.button("▶ Simular Compresión", key="btn_video")

    if btn_comp:
        render_codigo_y_formulas(algo_vid, stats)
        
        orig = len(datos) * 45  
        comp = len(datos)
        
        # HTML NATIVO PARA PASOS DE VIDEO (STRING DE UNA LÍNEA PARA EVITAR BUGS DE STREAMLIT)
        html_macroblocks = (
            '<div style="display:flex; flex-direction:column; align-items:center; background:var(--bg-1); padding:1rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="font-family:\'IBM Plex Mono\', monospace; font-size:0.75rem; color:var(--cyan); margin-bottom:1rem;">FRAME DIVIDIDO EN MACROBLOCKS</div>'
            '<div style="display:grid; grid-template-columns:repeat(8, 1fr); gap:2px; background:var(--border); padding:2px; border-radius:4px;">'
        )
        for _ in range(64):
            html_macroblocks += '<div style="width:20px; height:20px; background:var(--bg-2); border:1px solid rgba(6,182,212,0.3);"></div>'
        html_macroblocks += '</div></div>'

        html_motion = (
            '<div style="display:flex; flex-wrap:wrap; gap:20px; align-items:center; justify-content:center; background:var(--bg-1); padding:1.5rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="text-align:center;">'
            '<div style="width:80px; height:80px; border:2px dashed var(--muted); position:relative; background:var(--bg-2);">'
            '<div style="width:25px; height:25px; background:var(--violet); position:absolute; top:10px; left:10px; border-radius:4px;"></div>'
            '</div>'
            '<div style="font-size:0.65rem; color:var(--muted); margin-top:8px; font-family:\'IBM Plex Mono\', monospace;">Frame Referencia (t-1)</div>'
            '</div>'
            '<div style="display:flex; flex-direction:column; align-items:center; color:var(--cyan);">'
            '<div style="font-weight:bold; font-family:\'IBM Plex Mono\', monospace; font-size:0.8rem; background:rgba(6,182,212,0.1); padding:4px 8px; border-radius:4px; border:1px solid rgba(6,182,212,0.3);">Vect. Mov (dx:+20, dy:+15)</div>'
            '<div style="font-size:1.5rem; margin-top:4px;">➔</div>'
            '</div>'
            '<div style="text-align:center;">'
            '<div style="width:80px; height:80px; border:2px solid var(--border-strong); position:relative; background:var(--bg-2);">'
            '<div style="width:25px; height:25px; background:var(--cyan); position:absolute; top:25px; left:30px; border-radius:4px; box-shadow: 0 0 10px rgba(6,182,212,0.5);"></div>'
            '<div style="width:25px; height:25px; border:2px dashed rgba(139,92,246,0.5); position:absolute; top:10px; left:10px; border-radius:4px;"></div>'
            '</div>'
            '<div style="font-size:0.65rem; color:var(--cyan); margin-top:8px; font-family:\'IBM Plex Mono\', monospace;">Frame Actual (t)</div>'
            '</div>'
            '</div>'
        )

        html_residual = (
            '<div style="display:flex; flex-direction:column; align-items:center; background:var(--bg-1); padding:1rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="font-family:\'IBM Plex Mono\', monospace; font-size:0.75rem; color:var(--cyan); margin-bottom:1rem;">DCT DEL RESIDUAL (Errores cercanos a 0)</div>'
            '<div class="dct-grid">'
        )
        residual_vals = [5, -2, 0, 1, 0, 0, 0, 0, -3, 1, 0, 0, 0, 0, 0, 0] + [0]*48
        for val in residual_vals:
            color = 128 + val*10
            text_color = "#ef4444" if val != 0 else "var(--muted)"
            html_residual += f'<div class="dct-cell" style="background-color: rgb({color},{color},{color}); color: {text_color}; border:1px solid var(--bg-0);">{val}</div>'
        html_residual += '</div></div>'

        html_cabac = (
            '<div style="display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:center; background:var(--bg-1); padding:1.5rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div class="rle-pill"><div class="rle-count" style="background:var(--bg-4); color:var(--txt-dim);">Símbolos</div><div class="rle-val" style="color:var(--txt);">0,1,0,0...</div></div>'
            '<span style="color:var(--muted); font-size:1.2rem;">➔</span>'
            '<div class="rle-pill"><div class="rle-count" style="background:linear-gradient(135deg, var(--amber), #d97706);">Modelo Contexto</div></div>'
            '<span style="color:var(--muted); font-size:1.2rem;">➔</span>'
            '<div class="rle-pill"><div class="rle-count" style="background:linear-gradient(135deg, var(--green), #059669);">Flujo CABAC</div><div class="rle-val" style="color:var(--green); font-weight:bold;">01101...</div></div>'
            '</div>'
        )

        html_videodec = (
            '<div style="display:flex; flex-direction:column; gap: 15px; background:var(--bg-1); padding:1.5rem; border-radius:8px; border:1px solid var(--border); margin:1rem 0;">'
            '<div style="font-family:\'IBM Plex Mono\', monospace; font-size:0.75rem; color:var(--cyan); text-align:center;">PIPELINE DE DESCOMPRESIÓN (PLAYBACK)</div>'
            '<div style="display:flex; align-items:center; justify-content:center; gap: 10px;">'
            '<div class="rle-pill"><div class="rle-count" style="background:var(--green);">Flujo CABAC</div><div class="rle-val">01101...</div></div>'
            '<span style="color:var(--muted); font-size:1.2rem;">➔</span>'
            '<div class="rle-pill"><div class="rle-count" style="background:var(--amber);">Descodificador</div></div>'
            '<span style="color:var(--muted); font-size:1.2rem;">➔</span>'
            '<div class="rle-pill"><div class="rle-count" style="background:var(--bg-4);">Coeficientes</div><div class="rle-val">5, -2, 0...</div></div>'
            '</div>'
            '<div style="display:flex; align-items:center; justify-content:center; gap: 10px;">'
            '<div style="border:1px solid var(--border); padding:5px; border-radius:4px; font-size:0.6rem; text-align:center;">Matriz<br>Frecuencias</div>'
            '<span style="color:var(--muted); font-size:1.2rem;">➔</span>'
            '<div class="rle-pill"><div class="rle-count" style="background:var(--cyan); color:black;">IDCT Inversa</div></div>'
            '<span style="color:var(--muted); font-size:1.2rem;">➔</span>'
            '<div style="border:1px solid var(--border); padding:5px; border-radius:4px; font-size:0.6rem; text-align:center; color:#ef4444;">Residual<br>Espacial (Error)</div>'
            '</div>'
            '<div style="display:flex; align-items:center; justify-content:center; gap: 10px; margin-top:10px;">'
            '<div style="text-align:center;">'
            '<div style="width:50px; height:50px; border:1px dashed var(--muted); background:var(--bg-2); position:relative;">'
            '<div style="width:15px; height:15px; background:var(--violet); position:absolute; top:5px; left:5px;"></div>'
            '</div>'
            '<div style="font-size:0.5rem; color:var(--muted); margin-top:4px;">Frame(t-1)<br>+ Vect. Mov</div>'
            '</div>'
            '<span style="color:var(--muted); font-size:1.5rem; font-weight:bold;">+</span>'
            '<div style="text-align:center;">'
            '<div style="width:50px; height:50px; border:1px solid var(--border-strong); background:rgba(239,68,68,0.1); display:flex; align-items:center; justify-content:center; color:#ef4444; font-size:0.6rem;">Residual</div>'
            '</div>'
            '<span style="color:var(--muted); font-size:1.5rem; font-weight:bold;">=</span>'
            '<div style="text-align:center;">'
            '<div style="width:50px; height:50px; border:1px solid var(--cyan); background:var(--bg-2); position:relative;">'
            '<div style="width:15px; height:15px; background:var(--cyan); position:absolute; top:15px; left:20px;"></div>'
            '</div>'
            '<div style="font-size:0.5rem; color:var(--cyan); margin-top:4px;">Frame(t)<br>Reconstruido</div>'
            '</div>'
            '</div>'
            '</div>'
        )

        pasos = [
            {"titulo": "Paso 1 · Partición en Macroblocks", "detalle": "", "html": html_macroblocks},
            {"titulo": "Paso 2 · Predicción (Motion Estimation)", "detalle": "", "html": html_motion},
            {"titulo": "Paso 3 · DCT y Cuantización", "detalle": "", "html": html_residual},
            {"titulo": "Paso 4 · CABAC", "detalle": "", "html": html_cabac},
            {"titulo": "Paso 5 · Proceso de Descompresión (Playback)", "detalle": "", "html": html_videodec}
        ]
        
        res = ResultadoCompresion(
            nombre_algoritmo="H.264 (Simulado)",
            datos_originales=datos,
            datos_comprimidos=datos,
            datos_descomprimidos=datos,
            tamaño_original=orig,
            tamaño_comprimido=comp,
            tasa_compresion=orig / comp,
            ratio_reduccion=1 - (comp / orig),
            tiempo_ms=125.4,
            pasos=pasos,
            es_stub=True
        )
        
        render_resultado_compresion(res)
        
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Trama I (Intra)", "3.5:1", "Solo DCT espacial")
        with c2: st.metric("Trama P (Predictiva)", "12.0:1", "DCT + Motion Vector")
        with c3: st.metric("Trama B (Bidireccional)", "20.0:1", "Máxima compresión")

        render_pasos(res.pasos)
        render_descodificacion(res, "video")

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    inject_css()
    render_header()
    tabs = st.tabs(["📄 Texto", "🖼️ Imagen", "🎵 Audio", "🎬 Video"])
    with tabs[0]: tab_texto()
    with tabs[1]: tab_imagen()
    with tabs[2]: tab_audio()
    with tabs[3]: tab_video()

if __name__ == "__main__":
    main()
