from __future__ import annotations

import heapq
import io
import math
import struct
import time
import wave
import random
import base64
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
    page_title="DataLab · Compresión y Tx",
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
        
        /* BOTÓN DE DESCARGA NATIVO */
        .dl-btn { display: inline-block; background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white !important; padding: 0.55rem 2rem; border-radius: 10px; font-weight: 700; text-decoration: none; font-size: 0.85rem; font-family: 'Syne', sans-serif; box-shadow: 0 2px 12px rgba(16,185,129,0.2); transition: all 0.2s ease; margin-top: 10px; }
        .dl-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(16,185,129,0.4); text-decoration: none; }
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
#  INTEGRATION CLASSES (CHANNEL CODING & MODULATION)
# ═════════════════════════════════════════════════════════════════════════════

class MapeadorCanalModulacion:
    """Clase estática para simular la codificación de canal (FEC) y la modulación."""
    
    @staticmethod
    def simular_fec_interfaz(tasa: str, n_errores: int = 2) -> List[Dict[str, Any]]:
        """Genera el JSON Real (Log) de la corrección de errores para la Interfaz."""
        k, n = map(int, tasa.split('/'))
        log_fec = []
        for _ in range(5):
            bloque_orig = "".join(str(random.randint(0,1)) for _ in range(k))
            bloque_tx = bloque_orig + "".join(str(random.randint(0,1)) for _ in range(n-k))
            
            # Inyectar error simulado
            idx_err = random.randint(0, n-1)
            b_list = list(bloque_tx)
            b_list[idx_err] = '1' if b_list[idx_err] == '0' else '0'
            bloque_rx = "".join(b_list)
            
            log_fec.append({
                "Bloque Crudo (Fuente)": bloque_orig,
                "Bloque + Paridad (Tx)": bloque_tx,
                "Ruido Canal (Rx)": bloque_rx,
                "Síndrome Detectado": f"Error en bit {idx_err}",
                "Corrección (FEC)": f"Bit {idx_err} invertido",
                "Data Recuperada": bloque_orig
            })
        return log_fec

    @staticmethod
    def simular_constelacion_bpsk(ber: float) -> str:
        """Genera un SVG de constelación con dispersión basada en la Tasa de Error."""
        svg = '<svg viewBox="-150 -100 300 200" width="100%" height="200px" style="background:var(--bg-2); border-radius:8px; border:1px solid var(--border);">'
        svg += '<line x1="-150" y1="0" x2="150" y2="0" stroke="var(--muted)" stroke-width="1" />'
        svg += '<line x1="0" y1="-100" x2="0" y2="100" stroke="var(--muted)" stroke-width="1" />'
        
        # Centroides Ideales
        svg += '<circle cx="-80" cy="0" r="4" fill="var(--red)" />'
        svg += '<text x="-80" y="20" fill="var(--red)" font-family="monospace" font-size="10" text-anchor="middle">S0 (0)</text>'
        
        svg += '<circle cx="80" cy="0" r="4" fill="var(--cyan)" />'
        svg += '<text x="80" y="20" fill="var(--cyan)" font-family="monospace" font-size="10" text-anchor="middle">S1 (1)</text>'
        
        # Dispersión (AWGN)
        np.random.seed(int(time.time()))
        puntos = 150
        dispersión = 10 + (ber * 500) # El BER expande la nube
        
        for _ in range(puntos):
            # Nube del 0 (-80)
            x_noise = np.random.normal(-80, dispersión)
            y_noise = np.random.normal(0, dispersión)
            svg += f'<circle cx="{x_noise}" cy="{y_noise}" r="1.5" fill="var(--red)" opacity="0.4" />'
            
            # Nube del 1 (+80)
            x_noise = np.random.normal(80, dispersión)
            y_noise = np.random.normal(0, dispersión)
            svg += f'<circle cx="{x_noise}" cy="{y_noise}" r="1.5" fill="var(--cyan)" opacity="0.4" />'
            
        svg += '</svg>'
        return svg

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
            "    rankdir=RL;",            
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
            return s.replace("\\", "\\\\").replace('"', '\\"') if s else "Esp
