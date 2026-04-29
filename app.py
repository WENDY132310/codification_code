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

        /* ── CSS Variables ────────────────────────────────────── */
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
            --glow-cyan: 0 0 20px rgba(6,182,212,0.15);
            --glow-violet: 0 0 20px rgba(139,92,246,0.15);
            --radius-sm: 8px;
            --radius-md: 14px;
            --radius-lg: 20px;
        }

        /* ── Global Reset ─────────────────────────────────────── */
        html, body, [class*="css"], [data-testid] {
            font-family: 'Syne', sans-serif !important;
            background-color: var(--bg-0) !important;
            color: var(--txt) !important;
        }
        * { box-sizing: border-box; }

        /* ── Hide Streamlit Chrome ────────────────────────────── */
        #MainMenu, footer, header { visibility: hidden !important; }
        .stDeployButton, [data-testid="stToolbar"] { display: none !important; }
        section[data-testid="stSidebar"] { display: none !important; }

        /* ── Main Container ───────────────────────────────────── */
        .main .block-container {
            padding: 0 2rem 4rem !important;
            max-width: 1440px !important;
        }

        /* ── Scrollbar ────────────────────────────────────────── */
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: var(--bg-1); }
        ::-webkit-scrollbar-thumb { background: var(--cyan); border-radius: 3px; }

        /* ═══════════════════════════════════════════════════════
           APP HEADER
        ═══════════════════════════════════════════════════════ */
        .app-header-wrap {
            background: var(--bg-1);
            border-bottom: 1px solid var(--border);
            padding: 1.2rem 0 1.4rem;
            margin: 0 -2rem 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
        }
        .header-left { display: flex; align-items: center; gap: 1.25rem; }
        .header-icon {
            width: 48px; height: 48px;
            background: linear-gradient(135deg, var(--cyan), var(--violet));
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.4rem;
            box-shadow: var(--glow-cyan);
            flex-shrink: 0;
        }
        .app-title {
            font-size: 1.55rem;
            font-weight: 800;
            background: linear-gradient(100deg, var(--cyan) 0%, #a78bfa 60%, var(--violet) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.04em;
            line-height: 1.1;
            margin: 0;
        }
        .app-subtitle {
            font-family: 'IBM Plex Mono', monospace !important;
            color: var(--muted);
            font-size: 0.7rem;
            letter-spacing: 0.06em;
            margin-top: 0.2rem;
        }

        /* ═══════════════════════════════════════════════════════
           SECTION TITLES
        ═══════════════════════════════════════════════════════ */
        .section-label {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin: 1.5rem 0 0.85rem;
        }
        .section-label .sl-icon {
            font-size: 0.9rem;
        }
        .section-label .sl-text {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--cyan);
        }
        .section-label::after {
            content: '';
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, var(--border) 0%, transparent 100%);
        }

        /* ═══════════════════════════════════════════════════════
           ST.METRIC OVERRIDES
        ═══════════════════════════════════════════════════════ */
        [data-testid="stMetric"] {
            background: var(--bg-2) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            padding: 1.1rem 1.3rem !important;
            position: relative;
            overflow: hidden;
        }
        [data-testid="stMetric"]::before {
            content: '';
            position: absolute;
            bottom: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--violet), var(--cyan));
        }

        /* ═══════════════════════════════════════════════════════
           BUTTONS & UPLOADERS
        ═══════════════════════════════════════════════════════ */
        [data-testid="stFileUploader"] {
            background: var(--bg-2) !important;
            border: 1.5px dashed rgba(6,182,212,0.28) !important;
            border-radius: var(--radius-lg) !important;
            padding: 1.5rem !important;
            transition: border-color 0.25s !important;
        }
        .stButton > button {
            background: linear-gradient(135deg, var(--cyan) 0%, var(--violet) 100%) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            font-family: 'Syne', sans-serif !important;
            font-size: 0.85rem !important;
            padding: 0.55rem 2rem !important;
            letter-spacing: 0.03em !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 2px 12px rgba(6,182,212,0.2) !important;
        }

        /* ═══════════════════════════════════════════════════════
           CUSTOM HTML COMPONENTS
        ═══════════════════════════════════════════════════════ */
        .info-box {
            display: flex; gap: 0.9rem; align-items: flex-start;
            background: rgba(6,182,212,0.06); border: 1px solid rgba(6,182,212,0.22);
            border-radius: var(--radius-md); padding: 1rem 1.25rem; margin: 0.75rem 0;
        }
        .info-box.amber { background: rgba(245,158,11,0.06); border-color: rgba(245,158,11,0.22); }
        .info-box .ib-content { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.75rem; color: var(--txt); line-height: 1.7; }
        
        .dct-wrapper { display: flex; flex-direction: column; align-items: center; background: var(--bg-1); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border); margin: 1rem 0; }
        .dct-grid { display: grid; grid-template-columns: repeat(8, 1fr); gap: 2px; width: 100%; max-width: 320px; background: var(--border); padding: 2px; border-radius: 4px; }
        .dct-cell { aspect-ratio: 1; display: flex; align-items: center; justify-content: center; font-size: 0.65rem; font-family: 'IBM Plex Mono', monospace; border-radius: 2px; font-weight: 600; }
        
        .rle-wrapper { display: flex; flex-wrap: wrap; gap: 8px; background: var(--bg-1); padding: 1.25rem; border-radius: 8px; border: 1px solid var(--border); margin: 1rem 0; }
        .rle-pill { display: flex; border-radius: 6px; overflow: hidden; border: 1px solid var(--border); }
        .rle-count { background: linear-gradient(135deg, var(--cyan), var(--violet)); color: #fff; font-weight: 800; padding: 4px 10px; font-size: 0.8rem; font-family: 'IBM Plex Mono', monospace; display: flex; align-items: center; justify-content: center; }
        .rle-val { background: var(--bg-3); color: var(--txt); padding: 4px 12px; font-size: 0.8rem; font-family: 'IBM Plex Mono', monospace; display: flex; align-items: center; justify-content: center; }
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
    tamaño_original: int
    tamaño_comprimido: int
    tasa_compresion: float    
    ratio_reduccion: float    
    tiempo_ms: float
    pasos: List[Dict[str, Any]] = field(default_factory=list)
    tabla_codigos: Optional[Dict[str, str]] = None
    tabla_lzw: Optional[List[Dict]] = None
    grafo_dot: Optional[str] = None  # <--- Cambiamos de Mermaid a Graphviz DOT
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

# ═════════════════════════════════════════════════════════════════════════════
#  COMPRESSION ALGORITHMS
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

    # ── NUEVO: GENERADOR DE ÁRBOL USANDO GRAPHVIZ DOT (Nativo de Streamlit) ──
    def _generar_dot(self, nodo: Optional[NodoHuffman], total_simbolos: int) -> str:
        if not nodo: return ""
        
        lineas = [
            "digraph Huffman {",
            "    rankdir=RL;",           # Dibuja de Derecha a Izquierda (como en tu imagen)
            "    bgcolor=\"transparent\";",
            "    node [fontname=\"monospace\", style=\"filled\", fillcolor=\"#1e293b\", fontcolor=\"#cdd9e5\", color=\"#06b6d4\"];",
            "    edge [fontname=\"monospace\", fontcolor=\"#ef4444\", color=\"#8b9cb5\", fontsize=11, fontcolor=\"red\"];" # Numeros de rama en rojo
        ]
        contador = [0]

        def escapar_etiqueta(sym: Any) -> str:
            if isinstance(sym, int): return f"0x{sym:02X}"
            s = repr(sym)
            if s.startswith("'") and s.endswith("'"): s = s[1:-1]
            elif s.startswith('"') and s.endswith('"'): s = s[1:-1]
            s = s.replace("\\", "\\\\").replace('"', '\\"')
            return s if s else "Espacio"

        def recorrer(n: NodoHuffman) -> str:
            my_id = f"N{contador[0]}"
            contador[0] += 1
            prob = n.frecuencia / total_simbolos if total_simbolos > 0 else 0

            if n.simbolo is not None:
                simbolo_limpio = escapar_etiqueta(n.simbolo)
                # Nodo Hoja (Izquierda)
                etiqueta = f"{simbolo_limpio}: {prob:.2g}"
                lineas.append(f"    {my_id} [label=\"{etiqueta}\", shape=\"box\", fillcolor=\"#0f172a\", color=\"#06b6d4\"];")
            else:
                # Nodo Interno (Derecha)
                lineas.append(f"    {my_id} [label=\"{prob:.2g}\", shape=\"ellipse\", color=\"#8b5cf6\"];")

            if n.izquierda:
                izq_id = recorrer(n.izquierda)
                # Rama 0
                lineas.append(f"    {my_id} -> {izq_id} [label=\" 0 \", fontcolor=\"#06b6d4\"];")
            if n.derecha:
                der_id = recorrer(n.derecha)
                # Rama 1
                lineas.append(f"    {my_id} -> {der_id} [label=\" 1 \", fontcolor=\"#ef4444\"];")

            return my_id

        recorrer(nodo)
        lineas.append("}")
        return "\n".join(lineas)

    def comprimir(self) -> ResultadoCompresion:
        try: texto = self._datos.decode("utf-8", errors="replace")
        except Exception: texto = self._datos.decode("latin-1", errors="replace")
        
        freq = Counter(texto)
        raiz = self._construir_arbol(dict(freq))
        codigos = self._generar_codigos(raiz)
        
        total_bits = sum(len(codigos.get(c, "?")) for c in texto)
        comprimido = bytes(math.ceil(total_bits / 8)) 

        pasos = [
            {"titulo": "Paso 1", "detalle": f"Símbolos únicos: {len(freq)}"},
            {"titulo": "Paso 2", "detalle": "Fusión Bottom-Up usando Min-Heap."},
        ]

        grafo_dot = self._generar_dot(raiz, len(texto)) if len(freq) <= 60 else None

        return ResultadoCompresion("Huffman", self._datos, comprimido, len(self._datos), max(1, len(comprimido)), len(self._datos) / max(1, len(comprimido)), 1 - (max(1, len(comprimido)) / len(self._datos)), 10.0, pasos, codigos, grafo_dot=grafo_dot, es_stub=True)

# ─── B. LZW ──────────────────────────────────────────────────────────────────
class CodificadorLZW:
    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        texto = datos.decode("utf-8", errors="replace")[:10_000]
        dic, sig, w, codigos_salida, log_entradas = {chr(i): i for i in range(256)}, 256, "", [], []

        for i, c in enumerate(texto):
            wc = w + c
            if wc in dic: w = wc
            else:
                codigos_salida.append(dic[w])
                if len(log_entradas) < 20: log_entradas.append({"Pos": i, "w": repr(w), "c": repr(c), "Emitido": dic[w], "Nuevo": f"[{sig}]={repr(wc)}"})
                if sig < 4096: dic[wc] = sig; sig += 1
                w = c
        if w: codigos_salida.append(dic[w])

        comprimido = struct.pack(f">{len(codigos_salida)}H", *[min(c, 65535) for c in codigos_salida]) if codigos_salida else b""
        return ResultadoCompresion("LZW", datos, comprimido, len(datos), max(1, len(comprimido)), len(datos) / max(1, len(comprimido)), 1 - (max(1, len(comprimido)) / len(datos)), 10.0, [{"titulo": "Paso 1", "detalle": "Matching completado.", "tabla": log_entradas}], tabla_lzw=log_entradas, es_stub=False)

# ─── C. RLE ───────────────────────────────────────────────────────────────────
class CodificadorRLE:
    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        if not datos: return ResultadoCompresion("RLE", b"", b"", 0, 0, 1.0, 0.0, 0.0)
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

        pasos = [{"titulo": "Particionado Secuencial (Bloques RLE)", "detalle": "Datos agrupados:", "html": html_rle}]
        return ResultadoCompresion("RLE", datos, bytes(comprimido), len(datos), max(1, len(comprimido)), len(datos) / max(1, len(comprimido)), 1 - (max(1, len(comprimido)) / len(datos)), 10.0, pasos, es_stub=False)

# ─── D. DCT (Image / JPEG-like) ───────────────────────────────────────────────
class CodificadorDCT:
    def comprimir(self, datos: bytes, calidad: int = 50) -> ResultadoCompresion:
        raw = list(datos[:64]) + [128] * max(0, 64 - len(datos))
        html_dct = '<div class="dct-wrapper"><div class="dct-title">Bloque Extraído (8x8 px)</div><div class="dct-grid">'
        for val in raw[:64]:
            color = max(0, min(255, int(val)))
            text_color = "black" if color > 127 else "white"
            html_dct += f'<div class="dct-cell" style="background-color: rgb({color},{color},{color}); color: {text_color};">{color}</div>'
        html_dct += '</div></div>'
        
        pasos = [{"titulo": "Partición en Bloques", "detalle": "Bloque extraído en 8x8:", "html": html_dct}]
        return ResultadoCompresion("DCT", datos, bytes(max(1, len(datos)//2)), len(datos), max(1, len(datos)//2), 2.0, 0.5, 10.0, pasos, es_stub=True)

# ═════════════════════════════════════════════════════════════════════════════
#  UI HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════
def fmt_bytes(n: int) -> str:
    if n < 1024: return f"{n} B"
    if n < 1024**2: return f"{n/1024:.2f} KB"
    return f"{n/1024**2:.2f} MB"

def render_header() -> None:
    st.markdown('<div class="app-header-wrap"><div class="app-header"><div class="header-left"><div class="header-icon">⚡</div><div><p class="app-title">DataLab · Compresión</p></div></div></div></div>', unsafe_allow_html=True)

def section_label(icon: str, text: str) -> None:
    st.markdown(f'<div class="section-label"><span class="sl-icon">{icon}</span><span class="sl-text">{text}</span></div>', unsafe_allow_html=True)

def info_box(icon: str, content: str, variant: str = "") -> None:
    st.markdown(f'<div class="info-box {variant}"><div class="ib-icon">{icon}</div><div class="ib-content">{content}</div></div>', unsafe_allow_html=True)

def render_pasos(pasos: List[Dict[str, Any]]) -> None:
    section_label("🔍", "PROCEDIMIENTO PASO A PASO")
    for i, paso in enumerate(pasos, 1):
        with st.expander(paso.get("titulo", f"Paso {i}"), expanded=(i <= 2)):
            if paso.get("detalle"): st.markdown(f"<div style='font-size:0.8rem;color:var(--txt-dim);margin-bottom:10px;'>{paso['detalle']}</div>", unsafe_allow_html=True)
            if paso.get("html"): st.markdown(paso["html"], unsafe_allow_html=True)
            elif paso.get("tabla"): st.dataframe(pd.DataFrame(paso["tabla"]), use_container_width=True, height=250)

def render_arbol_huffman(res: ResultadoCompresion) -> None:
    if res.nombre_algoritmo == "Huffman":
        section_label("🌿", "DIAGRAMA DEL ÁRBOL DE HUFFMAN")
        if res.grafo_dot:
            info_box("💡", "Árbol generado con Graphviz (Natívo). La raíz está a la derecha y las hojas a la izquierda, indicando la probabilidad de cada nodo.")
            # Graphviz nativo de Streamlit, sin bugs de markdown
            st.graphviz_chart(res.grafo_dot)
        else:
            info_box("⚠️", "El archivo tiene muchos símbolos distintos, usa un texto más pequeño para visualizar el árbol.", "amber")

def render_no_file(icon: str, texto: str, subtexto: str) -> None:
    st.markdown(f'<div class="no-file-state"><div class="nfs-icon">{icon}</div><div class="nfs-title">{texto}</div><div class="nfs-sub">{subtexto}</div></div>', unsafe_allow_html=True)

def render_resultado_compresion(res: ResultadoCompresion) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Tamaño Original", fmt_bytes(res.tamaño_original))
    with c2: st.metric("Tamaño Comprimido", fmt_bytes(res.tamaño_comprimido))
    with c3: st.metric("Tasa", f"{res.tasa_compresion:.2f}:1")
    with c4: st.metric("Reducción", f"{max(0,res.ratio_reduccion)*100:.1f}%")

# ═════════════════════════════════════════════════════════════════════════════
#  TABS Y MAIN
# ═════════════════════════════════════════════════════════════════════════════

def tab_texto() -> None:
    uploaded = st.file_uploader("Sube un archivo de texto", type=["txt", "csv"], key="uploader_texto")
    if not uploaded: return render_no_file("📝", "Sube un archivo de texto", ".txt .csv")
    
    datos = uploaded.read()
    c_sel, c_btn = st.columns([3, 1], gap="medium")
    with c_sel: algo = st.selectbox("Algoritmo", ["Huffman", "LZW", "RLE"], key="algo_texto")
    with c_btn: comprimir = st.button("▶ Comprimir", key="btn_texto")

    if comprimir:
        if algo == "Huffman": res = CodificadorHuffman(datos).comprimir()
        elif algo == "LZW": res = CodificadorLZW().comprimir(datos)
        else: res = CodificadorRLE().comprimir(datos)

        render_resultado_compresion(res)
        render_pasos(res.pasos)
        render_arbol_huffman(res)

def tab_imagen() -> None:
    uploaded = st.file_uploader("Sube una imagen", type=["png", "jpg"], key="uploader_img")
    if not uploaded: return render_no_file("🖼️", "Sube una imagen", ".png .jpg")
    
    datos = uploaded.read()
    c_sel, c_btn = st.columns([3, 1], gap="medium")
    with c_sel: algo = st.selectbox("Algoritmo", ["DCT — JPEG-like", "RLE"], key="algo_img")
    with c_btn: comprimir = st.button("▶ Comprimir", key="btn_img")

    if comprimir:
        if algo == "DCT — JPEG-like": res = CodificadorDCT().comprimir(datos)
        else: res = CodificadorRLE().comprimir(datos)
        
        render_resultado_compresion(res)
        render_pasos(res.pasos)

def main() -> None:
    inject_css()
    render_header()
    tabs = st.tabs(["📄 Texto", "🖼️ Imagen"])
    with tabs[0]: tab_texto()
    with tabs[1]: tab_imagen()

if __name__ == "__main__":
    main()
