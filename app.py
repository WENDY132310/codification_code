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
    """
    Inject custom CSS for the dark 'lab terminal' aesthetic.
    Uses IBM Plex Mono (mono) + Syne (display) as font pair.
    Palette: #030712 bg · #06b6d4 cyan · #8b5cf6 violet
    """
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
        .header-badges { display: flex; gap: 0.4rem; flex-wrap: wrap; align-items: center; }
        .hbadge {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.6rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            border: 1px solid;
        }
        .hbadge-c { color: var(--cyan); border-color: var(--border-strong); background: var(--cyan-dim); }
        .hbadge-v { color: var(--violet); border-color: rgba(139,92,246,0.3); background: var(--violet-dim); }
        .hbadge-g { color: var(--green); border-color: rgba(16,185,129,0.3); background: rgba(16,185,129,0.1); }

        /* ═══════════════════════════════════════════════════════
           TABS
        ═══════════════════════════════════════════════════════ */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--bg-2) !important;
            border-radius: var(--radius-md) !important;
            padding: 5px !important;
            gap: 4px !important;
            border: 1px solid var(--border) !important;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            border-radius: var(--radius-sm) !important;
            color: var(--muted) !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            padding: 0.55rem 1.4rem !important;
            border: none !important;
            transition: all 0.25s ease !important;
            letter-spacing: 0.02em !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--cyan-dim), var(--violet-dim)) !important;
            color: var(--cyan) !important;
            border: 1px solid var(--border-strong) !important;
        }
        .stTabs [data-baseweb="tab-panel"] { padding: 1.75rem 0 0 !important; }
        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] { display: none !important; }

        /* ═══════════════════════════════════════════════════════
           CARDS & CONTAINERS
        ═══════════════════════════════════════════════════════ */
        .card {
            background: var(--bg-2);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            position: relative;
            overflow: hidden;
        }
        .card-glow::after {
            content: '';
            position: absolute;
            top: -40px; right: -40px;
            width: 120px; height: 120px;
            background: radial-gradient(circle, rgba(6,182,212,0.08) 0%, transparent 70%);
            pointer-events: none;
        }
        .card-accent {
            background: var(--bg-2);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            position: relative;
            overflow: hidden;
        }
        .card-accent::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--cyan) 0%, var(--violet) 100%);
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
        [data-testid="stMetricLabel"] > div {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.62rem !important;
            color: var(--muted) !important;
            letter-spacing: 0.1em !important;
            text-transform: uppercase !important;
        }
        [data-testid="stMetricValue"] > div {
            font-size: 1.7rem !important;
            font-weight: 700 !important;
            color: var(--cyan) !important;
            font-family: 'IBM Plex Mono', monospace !important;
            letter-spacing: -0.03em !important;
        }
        [data-testid="stMetricDelta"] {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.7rem !important;
            color: var(--txt-dim) !important;
        }
        [data-testid="stMetricDelta"] svg { display: none !important; }

        /* ═══════════════════════════════════════════════════════
           FILE UPLOADER
        ═══════════════════════════════════════════════════════ */
        [data-testid="stFileUploader"] {
            background: var(--bg-2) !important;
            border: 1.5px dashed rgba(6,182,212,0.28) !important;
            border-radius: var(--radius-lg) !important;
            padding: 1.5rem !important;
            transition: border-color 0.25s !important;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: var(--cyan) !important;
            box-shadow: var(--glow-cyan) !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: transparent !important;
        }

        /* ═══════════════════════════════════════════════════════
           EXPANDER
        ═══════════════════════════════════════════════════════ */
        details[data-testid="stExpander"] {
            background: var(--bg-2) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            overflow: hidden;
        }
        details[data-testid="stExpander"] summary {
            background: var(--bg-3) !important;
            color: var(--cyan) !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.78rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.04em !important;
            padding: 0.75rem 1rem !important;
        }
        details[data-testid="stExpander"][open] summary {
            border-bottom: 1px solid var(--border) !important;
        }

        /* ═══════════════════════════════════════════════════════
           CODE BLOCKS
        ═══════════════════════════════════════════════════════ */
        .stCode, [data-testid="stCode"] {
            background: #010508 !important;
            border: 1px solid rgba(6,182,212,0.18) !important;
            border-radius: var(--radius-md) !important;
        }
        code { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important; }

        /* ═══════════════════════════════════════════════════════
           DATAFRAME
        ═══════════════════════════════════════════════════════ */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            overflow: hidden !important;
        }

        /* ═══════════════════════════════════════════════════════
           BUTTONS
        ═══════════════════════════════════════════════════════ */
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
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 24px rgba(6,182,212,0.35) !important;
        }

        /* ═══════════════════════════════════════════════════════
           PROGRESS BAR
        ═══════════════════════════════════════════════════════ */
        [data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, var(--cyan) 0%, var(--violet) 100%) !important;
            border-radius: 999px !important;
        }
        [data-testid="stProgressBar"] > div {
            background: var(--bg-3) !important;
            border-radius: 999px !important;
            height: 6px !important;
        }

        /* ═══════════════════════════════════════════════════════
           SELECTBOX
        ═══════════════════════════════════════════════════════ */
        [data-testid="stSelectbox"] > div > div {
            background: var(--bg-2) !important;
            border: 1px solid var(--border-strong) !important;
            border-radius: 10px !important;
            color: var(--txt) !important;
        }

        /* ═══════════════════════════════════════════════════════
           CUSTOM HTML COMPONENTS
        ═══════════════════════════════════════════════════════ */
        .info-box {
            display: flex;
            gap: 0.9rem;
            align-items: flex-start;
            background: rgba(6,182,212,0.06);
            border: 1px solid rgba(6,182,212,0.22);
            border-radius: var(--radius-md);
            padding: 1rem 1.25rem;
            margin: 0.75rem 0;
        }
        .info-box.violet {
            background: rgba(139,92,246,0.06);
            border-color: rgba(139,92,246,0.22);
        }
        .info-box.amber {
            background: rgba(245,158,11,0.06);
            border-color: rgba(245,158,11,0.22);
        }
        .info-box .ib-icon { font-size: 1.05rem; line-height: 1.5; }
        .info-box .ib-content {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.75rem;
            color: var(--txt);
            line-height: 1.7;
        }
        .info-box .ib-content strong { color: var(--cyan); }
        .info-box.violet .ib-content strong { color: var(--violet); }
        .info-box.amber  .ib-content strong { color: var(--amber); }

        .formula-display {
            background: var(--bg-0);
            border: 1px solid var(--border);
            border-left: 3px solid var(--cyan);
            border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
            padding: 0.75rem 1.25rem;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.82rem;
            color: var(--cyan);
            margin: 0.4rem 0;
        }

        .step-row {
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
            margin-bottom: 0.85rem;
        }
        .step-num {
            min-width: 30px; height: 30px;
            background: linear-gradient(135deg, var(--cyan), var(--violet));
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.68rem; font-weight: 700;
            color: #fff;
            font-family: 'IBM Plex Mono', monospace;
            flex-shrink: 0;
            margin-top: 0.1rem;
            box-shadow: 0 2px 8px rgba(6,182,212,0.3);
        }
        .step-body {
            flex: 1;
            background: var(--bg-3);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.7rem 1rem;
        }
        .step-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--cyan);
            margin-bottom: 0.2rem;
        }
        .step-detail {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            color: var(--txt-dim);
            line-height: 1.65;
        }

        .algo-card {
            background: var(--bg-3);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 1.1rem 1.25rem;
            margin-bottom: 0.75rem;
            position: relative;
            overflow: hidden;
        }
        .algo-card .ac-top {
            display: flex; align-items: center;
            justify-content: space-between;
            margin-bottom: 0.4rem;
        }
        .algo-card .ac-name {
            font-weight: 700;
            font-size: 0.9rem;
            color: var(--txt);
        }
        .algo-card .ac-tag {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.6rem;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
        }
        .algo-card .ac-desc {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            color: var(--muted);
            line-height: 1.55;
        }
        .tag-c { color: var(--cyan); background: var(--cyan-dim); border: 1px solid var(--border-strong); }
        .tag-v { color: var(--violet); background: var(--violet-dim); border: 1px solid rgba(139,92,246,0.3); }
        .tag-g { color: var(--green); background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.25); }
        .tag-a { color: var(--amber); background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.25); }

        .result-bar-wrap {
            background: var(--bg-3);
            border-radius: 999px;
            height: 8px; width: 100%;
            overflow: hidden; margin-top: 0.5rem;
        }
        .result-bar-fill {
            height: 100%; border-radius: 999px;
            background: linear-gradient(90deg, var(--cyan), var(--violet));
            transition: width 0.6s ease;
        }

        .huffman-code-row {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: var(--bg-3);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.2rem 0.5rem;
            margin: 2px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem;
        }
        .huffman-sym { color: var(--txt); font-weight: 600; }
        .huffman-bits { color: var(--cyan); }
        .huffman-len { color: var(--muted); }

        .stat-mini {
            display: flex; flex-direction: column;
            background: var(--bg-3);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.75rem 1rem;
            text-align: center;
        }
        .stat-mini .sm-val {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.3rem; font-weight: 600;
            color: var(--cyan);
        }
        .stat-mini .sm-lbl {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.6rem; color: var(--muted);
            text-transform: uppercase; letter-spacing: 0.1em;
            margin-top: 0.1rem;
        }

        .entropy-bar-bg {
            background: var(--bg-3);
            border-radius: 999px;
            height: 10px; width: 100%; overflow: hidden;
            margin: 0.4rem 0;
        }
        .entropy-bar-fill {
            height: 100%; border-radius: 999px;
            background: linear-gradient(90deg, var(--green) 0%, var(--cyan) 60%, var(--violet) 100%);
        }

        .no-file-state {
            text-align: center;
            padding: 3rem 2rem;
            background: var(--bg-2);
            border: 1.5px dashed var(--border);
            border-radius: var(--radius-lg);
            margin-top: 1.5rem;
        }
        .no-file-state .nfs-icon {
            font-size: 2.5rem; margin-bottom: 0.75rem;
        }
        .no-file-state .nfs-title {
            font-size: 1rem; font-weight: 600;
            color: var(--txt-dim); margin-bottom: 0.4rem;
        }
        .no-file-state .nfs-sub {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem; color: var(--muted);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class EstadisticasInfo:
    """All information-theory statistics computed from a byte sequence."""
    frecuencias: Dict[str, int]
    probabilidades: Dict[str, float]
    entropia: float           # H(X) in bits/symbol
    longitud_promedio: float  # L̄ = Σ pᵢ·lᵢ
    eficiencia: float         # η = H / L̄
    redundancia: float        # R = 1 − η
    total_simbolos: int
    simbolos_unicos: int
    entropia_maxima: float    # log₂(N) — upper bound


@dataclass
class ResultadoCompresion:
    """Typed result container from any compression algorithm."""
    nombre_algoritmo: str
    datos_originales: bytes
    datos_comprimidos: bytes
    tamaño_original: int
    tamaño_comprimido: int
    tasa_compresion: float    # original / compressed
    ratio_reduccion: float    # 1 − (comp/orig)
    tiempo_ms: float
    pasos: List[Dict[str, Any]] = field(default_factory=list)
    tabla_codigos: Optional[Dict[str, str]] = None
    tabla_lzw: Optional[List[Dict]] = None
    grafo_dot: Optional[str] = None  # <--- NUEVO CAMPO PARA EL ÁRBOL
    es_stub: bool = True


# ═════════════════════════════════════════════════════════════════════════════
#  ABSTRACT BASE CLASS — Information Analyzer
# ═════════════════════════════════════════════════════════════════════════════

class AnalizadorInformacion(ABC):
    """
    Abstract base for Shannon information-theory analysis.

    Subclasses implement ``_extraer_simbolos()`` to define the
    fundamental unit of information (character, byte, pixel, sample).

    Core formulas:
        H(X) = −∑ p(xᵢ)·log₂p(xᵢ)          (entropy)
        L̄    =  ∑ p(xᵢ)·lᵢ                 (avg code length)
        η    = H(X) / L̄                      (efficiency)
    """

    def __init__(self, datos: bytes, nombre: str = "datos") -> None:
        if not datos:
            raise ValueError("Los datos no pueden estar vacíos.")
        self._datos = datos
        self._nombre = nombre

    @abstractmethod
    def _extraer_simbolos(self) -> List[Any]:
        """Define the symbol alphabet for this data type."""
        ...

    # ── Public API ───────────────────────────────────────────────────────────

    def calcular_frecuencias(self) -> Dict[str, int]:
        """Count occurrences of each symbol."""
        simbolos = self._extraer_simbolos()
        return dict(Counter(simbolos))

    def calcular_probabilidades(self) -> Dict[str, float]:
        """p(xᵢ) = freq(xᵢ) / N"""
        freq = self.calcular_frecuencias()
        total = sum(freq.values())
        return {k: v / total for k, v in freq.items()}

    def calcular_entropia(self) -> float:
        """H(X) = −∑ p(xᵢ)·log₂p(xᵢ)  [bits/symbol]"""
        probs = self.calcular_probabilidades()
        return -sum(p * math.log2(p) for p in probs.values() if p > 0)

    def calcular_longitud_promedio(
        self, longitudes: Optional[Dict[str, int]] = None
    ) -> float:
        """
        L̄ = ∑ pᵢ·lᵢ

        Args:
            longitudes: symbol→code_length mapping.
                        If None, uses Huffman bound: lᵢ ≈ ⌈−log₂(pᵢ)⌉
        """
        probs = self.calcular_probabilidades()
        if longitudes:
            return sum(
                probs[s] * longitudes[s]
                for s in probs
                if s in longitudes
            )
        return sum(
            p * math.ceil(-math.log2(p))
            for p in probs.values()
            if p > 0
        )

    def calcular_eficiencia(
        self, L_bar: Optional[float] = None
    ) -> float:
        """η = H(X) / L̄"""
        H = self.calcular_entropia()
        L = L_bar if L_bar is not None else self.calcular_longitud_promedio()
        return H / L if L > 0 else 0.0

    def calcular_todo(self) -> EstadisticasInfo:
        """Run full pipeline and return a populated EstadisticasInfo."""
        freq = self.calcular_frecuencias()
        probs = self.calcular_probabilidades()
        H = self.calcular_entropia()
        L = self.calcular_longitud_promedio()
        eta = self.calcular_eficiencia(L)
        n = len(freq)
        return EstadisticasInfo(
            frecuencias=freq,
            probabilidades=probs,
            entropia=H,
            longitud_promedio=L,
            eficiencia=eta,
            redundancia=1.0 - eta,
            total_simbolos=sum(freq.values()),
            simbolos_unicos=n,
            entropia_maxima=math.log2(n) if n > 1 else 0.0,
        )

    @property
    def tamaño_bytes(self) -> int:
        return len(self._datos)


# ── Concrete Analyzers ────────────────────────────────────────────────────────

class AnalizadorTexto(AnalizadorInformacion):
    """Symbol unit = UTF-8 character."""

    def _extraer_simbolos(self) -> List[str]:
        try:
            return list(self._datos.decode("utf-8", errors="replace"))
        except Exception:
            return list(self._datos.decode("latin-1", errors="replace"))


class AnalizadorImagen(AnalizadorInformacion):
    """Symbol unit = individual byte value (0–255)."""

    def _extraer_simbolos(self) -> List[int]:
        return list(self._datos)


class AnalizadorAudio(AnalizadorInformacion):
    """Symbol unit = PCM sample byte."""

    def _extraer_simbolos(self) -> List[int]:
        return list(self._datos)


class AnalizadorVideo(AnalizadorInformacion):
    """Symbol unit = raw byte (capped at 100 KB for performance)."""

    CAP = 100_000

    def _extraer_simbolos(self) -> List[int]:
        return list(self._datos[: self.CAP])


# ═════════════════════════════════════════════════════════════════════════════
#  COMPRESSION ALGORITHMS
# ═════════════════════════════════════════════════════════════════════════════

# ─── A. HUFFMAN CODING ───────────────────────────────────────────────────────

class NodoHuffman:
    """Node in a Huffman binary tree."""

    __slots__ = ("simbolo", "frecuencia", "izquierda", "derecha")

    def __init__(self, simbolo: Optional[str], frecuencia: int) -> None:
        self.simbolo = simbolo
        self.frecuencia = frecuencia
        self.izquierda: Optional["NodoHuffman"] = None
        self.derecha: Optional["NodoHuffman"] = None

    def __lt__(self, other: "NodoHuffman") -> bool:  # for heapq
        return self.frecuencia < other.frecuencia


class CodificadorHuffman:
    """
    Huffman entropy coding implementation.
    """

    def __init__(self, datos: bytes) -> None:
        self._datos = datos

    def _construir_arbol(
        self, frecuencias: Dict[str, int]
    ) -> Optional[NodoHuffman]:
        """Greedy min-heap merge — O(n log n)."""
        heap: List[NodoHuffman] = [
            NodoHuffman(s, f) for s, f in frecuencias.items()
        ]
        heapq.heapify(heap)
        while len(heap) > 1:
            izq = heapq.heappop(heap)
            der = heapq.heappop(heap)
            padre = NodoHuffman(None, izq.frecuencia + der.frecuencia)
            padre.izquierda = izq
            padre.derecha = der
            heapq.heappush(heap, padre)
        return heap[0] if heap else None

    def _generar_codigos(
        self,
        nodo: Optional[NodoHuffman],
        prefijo: str = "",
        codigos: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """DFS traversal: left='0', right='1'."""
        if codigos is None:
            codigos = {}
        if nodo is None:
            return codigos
        if nodo.simbolo is not None:
            codigos[nodo.simbolo] = prefijo if prefijo else "0"
        else:
            self._generar_codigos(nodo.izquierda, prefijo + "0", codigos)
            self._generar_codigos(nodo.derecha, prefijo + "1", codigos)
        return codigos

    # ── NUEVO: GENERADOR DE ÁRBOL USANDO GRAPHVIZ DOT (Igual a la imagen) ──
    def _generar_dot(self, nodo: Optional[NodoHuffman], total_simbolos: int) -> str:
        if not nodo: return ""
        
        lineas = [
            "digraph Huffman {",
            "    rankdir=RL;",           # Dibuja de Derecha a Izquierda
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
            s = s.replace("\\", "\\\\").replace('"', '\\"')
            return s if s else "Espacio"

        def recorrer(n: NodoHuffman) -> str:
            my_id = f"N{contador[0]}"
            contador[0] += 1
            prob = n.frecuencia / total_simbolos if total_simbolos > 0 else 0

            if n.simbolo is not None:
                simbolo_limpio = escapar_etiqueta(n.simbolo)
                # Nodo Hoja: "simbolo: probabilidad"
                etiqueta = f"{simbolo_limpio}: {prob:.2g}"
                lineas.append(f"    {my_id} [label=\"{etiqueta}\", shape=\"box\", fillcolor=\"#0f172a\", color=\"#06b6d4\"];")
            else:
                # Nodo Interno: Probabilidad acumulada
                lineas.append(f"    {my_id} [label=\"{prob:.2g}\", shape=\"ellipse\", color=\"#8b5cf6\"];")

            if n.izquierda:
                izq_id = recorrer(n.izquierda)
                lineas.append(f"    {my_id} -> {izq_id} [label=\" 0 \", fontcolor=\"#06b6d4\"];")
            if n.derecha:
                der_id = recorrer(n.derecha)
                lineas.append(f"    {my_id} -> {der_id} [label=\" 1 \", fontcolor=\"#ef4444\"];")

            return my_id

        recorrer(nodo)
        lineas.append("}")
        return "\n".join(lineas)

    def _comprimir_bytes(self, codigos: Dict[str, str]) -> bytes:
        """STUB — theoretical size."""
        try:
            texto = self._datos.decode("utf-8", errors="replace")
        except Exception:
            texto = "".join(chr(b) for b in self._datos[:10_000])

        total_bits = sum(len(codigos.get(c, "?")) for c in texto)
        theo_bytes = math.ceil(total_bits / 8)
        return bytes(theo_bytes) 

    def comprimir(self) -> ResultadoCompresion:
        t0 = time.perf_counter()
        try:
            texto = self._datos.decode("utf-8", errors="replace")
        except Exception:
            texto = self._datos.decode("latin-1", errors="replace")

        freq = Counter(texto)
        raiz = self._construir_arbol(dict(freq))
        codigos = self._generar_codigos(raiz)
        
        comprimido = self._comprimir_bytes(codigos)
        
        pasos = [
            {"titulo": "Análisis de Frecuencias", "detalle": f"Símbolos únicos: {len(freq)}", "tabla": None},
            {"titulo": "Construcción del Árbol Huffman", "detalle": "Fusión Bottom-Up usando Min-Heap.", "tabla": None},
        ]

        # Generar el grafo limitando la cantidad para rendimiento
        grafo_dot = self._generar_dot(raiz, len(texto)) if len(freq) <= 60 else None

        t1 = time.perf_counter()
        orig = len(self._datos)
        comp = max(1, len(comprimido))

        return ResultadoCompresion(
            nombre_algoritmo="Huffman",
            datos_originales=self._datos,
            datos_comprimidos=comprimido,
            tamaño_original=orig,
            tamaño_comprimido=comp,
            tasa_compresion=orig / comp,
            ratio_reduccion=1 - (comp / orig),
            tiempo_ms=(t1 - t0) * 1000,
            pasos=pasos,
            tabla_codigos=codigos,
            grafo_dot=grafo_dot,
            es_stub=True,
        )


# ─── B. LZW ──────────────────────────────────────────────────────────────────

class CodificadorLZW:
    DICT_MAX = 4096 

    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        t0 = time.perf_counter()
        try:
            texto = datos.decode("utf-8", errors="replace")
        except Exception:
            texto = datos.decode("latin-1", errors="replace")
        texto = texto[:10_000]

        diccionario: Dict[str, int] = {chr(i): i for i in range(256)}
        siguiente = 256
        w = ""
        codigos_salida: List[int] = []
        log_entradas: List[Dict] = []

        for i, c in enumerate(texto):
            wc = w + c
            if wc in diccionario:
                w = wc
            else:
                codigos_salida.append(diccionario[w])
                if len(log_entradas) < 20:
                    log_entradas.append({"Pos": i, "Buffer w": repr(w), "Char c": repr(c), "Código": diccionario[w], "Nuevo": f"[{siguiente}]={repr(wc)}"})
                if siguiente < self.DICT_MAX:
                    diccionario[wc] = siguiente
                    siguiente += 1
                w = c

        if w: codigos_salida.append(diccionario[w])
        comprimido = struct.pack(f">{len(codigos_salida)}H", *[min(c, 65535) for c in codigos_salida]) if codigos_salida else b""
        
        t1 = time.perf_counter()
        return ResultadoCompresion("LZW", datos, comprimido, len(datos), max(1, len(comprimido)), len(datos)/max(1, len(comprimido)), 1-(len(comprimido)/len(datos)), (t1-t0)*1000, [{"titulo": "Matching", "detalle": "Completado", "tabla": log_entradas}], tabla_lzw=log_entradas, es_stub=False)


# ─── C. RLE ───────────────────────────────────────────────────────────────────

class CodificadorRLE:
    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        t0 = time.perf_counter()
        if not datos: return ResultadoCompresion("RLE", b"", b"", 0, 0, 1.0, 0.0, 0.0)
        comprimido = bytearray()
        runs_log: List[Dict] = []
        i = 0
        while i < len(datos):
            byte_actual = datos[i]
            conteo = 1
            while i+conteo < len(datos) and datos[i+conteo] == byte_actual and conteo < 255:
                conteo += 1
            comprimido.extend([conteo, byte_actual])
            if len(runs_log) < 25:
                runs_log.append({"Pos": i, "Byte": f"0x{byte_actual:02X}", "Run": conteo})
            i += conteo
        t1 = time.perf_counter()
        return ResultadoCompresion("RLE", datos, bytes(comprimido), len(datos), len(comprimido), len(datos)/max(1, len(comprimido)), 1-(len(comprimido)/len(datos)), (t1-t0)*1000, [{"titulo": "Escaneo", "detalle": "Runs encontrados", "tabla": runs_log}], es_stub=False)


# ─── D. DCT (Image / JPEG-like) ───────────────────────────────────────────────

class CodificadorDCT:
    Q_LUMA = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]], dtype=float)

    def _dct2d_manual(self, bloque: np.ndarray) -> np.ndarray:
        N = 8
        b = bloque.astype(float) - 128.0
        F = np.zeros((N, N))
        for u in range(N):
            for v in range(N):
                cu = 1.0 / math.sqrt(2) if u == 0 else 1.0
                cv = 1.0 / math.sqrt(2) if v == 0 else 1.0
                xs, ys = np.arange(N), np.arange(N)
                cos_u = np.cos((2*xs+1)*u*math.pi/(2*N))
                cos_v = np.cos((2*ys+1)*v*math.pi/(2*N))
                F[u,v] = (2/N)*cu*cv*np.sum(b * cos_u[:, None] * cos_v[None, :])
        return F

    def comprimir(self, datos: bytes, calidad: int = 50) -> ResultadoCompresion:
        t0 = time.perf_counter()
        raw = list(datos[:64]) + [128] * max(0, 64 - len(datos))
        bloque = np.array(raw[:64], dtype=float).reshape(8, 8)
        F = self._dct2d_manual(bloque)
        t1 = time.perf_counter()
        return ResultadoCompresion("DCT", datos, bytes(32), len(datos), 32, 2.0, 0.5, (t1-t0)*1000, [{"titulo": "DCT", "detalle": "Bloque 8x8 procesado", "tabla": None}], es_stub=True)


# ─── E. μ-Law (Audio) ────────────────────────────────────────────────────────

class CodificadorMuLaw:
    MU, BIAS = 255, 132
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
        t1 = time.perf_counter()
        return ResultadoCompresion("μ-Law", datos, encoded, len(datos), len(encoded), 2.0, 0.5, (t1-t0)*1000, [{"titulo": "G.711", "detalle": "Companding 2:1", "tabla": None}], es_stub=False)


# ─── F. ADPCM (Audio — Stub) ─────────────────────────────────────────────────

class CodificadorADPCM:
    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        return ResultadoCompresion("ADPCM", datos, bytes(len(datos)//4), len(datos), len(datos)//4, 4.0, 0.75, 1.0, [{"titulo": "IMA", "detalle": "Stub", "tabla": None}], es_stub=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SOURCE CODE STRINGS
# ═════════════════════════════════════════════════════════════════════════════
_CODE_ENTROPIA = """def entropia_shannon(datos):\n    freq = Counter(datos)\n    N = sum(freq.values())\n    return -sum((f/N)*math.log2(f/N) for f in freq.values() if f > 0)"""
_CODE_HUFFMAN = "# Huffman logic..."
_CODE_LZW = "# LZW logic..."
_CODE_RLE = "# RLE logic..."
_CODE_DCT = "# DCT logic..."
_CODE_MULAW = "# Mu-Law logic..."
_CODE_ADPCM = "# ADPCM logic..."


# ═════════════════════════════════════════════════════════════════════════════
#  UI HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def fmt_bytes(n: int) -> str:
    if n < 1024: return f"{n} B"
    if n < 1024**2: return f"{n/1024:.2f} KB"
    return f"{n/1024**2:.2f} MB"

def render_header() -> None:
    st.markdown('<div class="app-header-wrap"><div class="app-header"><div class="header-left"><div class="header-icon">⚡</div><div><p class="app-title">DataLab · Compresión</p><p class="app-subtitle">Shannon · Huffman · LZW · RLE · DCT · μ-Law · ADPCM</p></div></div></div></div>', unsafe_allow_html=True)

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

def render_tabla_simbolos(stats: EstadisticasInfo) -> None:
    rows = []
    for sym, freq in sorted(stats.frecuencias.items(), key=lambda x: -x[1])[:35]:
        prob = stats.probabilidades[sym]
        rows.append({"Símbolo": repr(sym), "Frecuencia": freq, "Probabilidad p(x)": round(prob, 6)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=350)

def render_resultado_compresion(res: ResultadoCompresion) -> None:
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
            if paso.get("detalle"): st.markdown(f"<div>{paso['detalle']}</div>", unsafe_allow_html=True)
            if paso.get("tabla"): st.dataframe(pd.DataFrame(paso["tabla"]), use_container_width=True)

# ── NUEVO: VISUALIZADOR DE ÁRBOL NATIVO ──
def render_arbol_huffman(res: ResultadoCompresion) -> None:
    if res.nombre_algoritmo == "Huffman" and res.grafo_dot:
        section_label("🌿", "DIAGRAMA DEL ÁRBOL DE HUFFMAN")
        info_box("💡", "Árbol horizontal (Derecha a Izquierda) con probabilidades y etiquetas de bit.")
        st.graphviz_chart(res.grafo_dot)

def render_no_file(icon: str, texto: str, subtexto: str) -> None:
    st.markdown(f'<div class="no-file-state"><div class="nfs-icon">{icon}</div><div class="nfs-title">{texto}</div><div class="nfs-sub">{subtexto}</div></div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB IMPLEMENTATIONS
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
    render_tabla_simbolos(stats)

    section_label("⚙️", "ALGORITMO DE COMPRESIÓN")
    c_sel, c_btn = st.columns([3, 1])
    with c_sel: algo = st.selectbox("Algoritmo", ["Huffman", "LZW", "RLE"], key="algo_texto")
    with c_btn: btn_comp = st.button("▶ Comprimir", key="btn_texto")

    if btn_comp:
        if algo == "Huffman": res = CodificadorHuffman(datos).comprimir()
        elif algo == "LZW": res = CodificadorLZW().comprimir(datos)
        else: res = CodificadorRLE().comprimir(datos)

        render_resultado_compresion(res)
        render_pasos(res.pasos)
        render_arbol_huffman(res) # <--- LLAMADA AL ÁRBOL


def tab_imagen() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")
    with col_up:
        section_label("🖼️", "CARGAR IMAGEN")
        uploaded = st.file_uploader("Sube una imagen", type=["png", "jpg"], key="uploader_imagen")
    
    if uploaded is None:
        render_no_file("🖼️", "Sube una imagen", ".png .jpg")
        return

    datos = uploaded.read()
    pc1, pc2 = st.columns([1, 2], gap="large")
    with pc1: st.image(datos, use_container_width=True)
    with pc2:
        analizador = AnalizadorImagen(datos, uploaded.name)
        stats = analizador.calcular_todo()
        render_dashboard(stats)

    algo_img = st.selectbox("Algoritmo", ["DCT — JPEG-like", "RLE", "Huffman (bytes)"], key="algo_imagen")
    btn_comp = st.button("▶ Comprimir", key="btn_imagen")

    if btn_comp:
        if algo_img == "DCT — JPEG-like": res = CodificadorDCT().comprimir(datos)
        elif algo_img == "RLE": res = CodificadorRLE().comprimir(datos)
        else: res = CodificadorHuffman(datos).comprimir()
        render_resultado_compresion(res)
        render_pasos(res.pasos)
        render_arbol_huffman(res)

def tab_audio() -> None:
    uploaded = st.file_uploader("Sube audio (.wav)", type=["wav"], key="uploader_audio")
    if not uploaded: return
    datos = uploaded.read()
    analizador = AnalizadorAudio(datos, uploaded.name)
    render_dashboard(analizador.calcular_todo())
    algo = st.selectbox("Algoritmo", ["μ-Law G.711", "ADPCM", "Huffman (bytes)"], key="algo_audio")
    if st.button("▶ Comprimir", key="btn_audio"):
        if algo == "μ-Law G.711": res = CodificadorMuLaw().comprimir(datos)
        elif algo == "ADPCM": res = CodificadorADPCM().comprimir(datos)
        else: res = CodificadorHuffman(datos).comprimir()
        render_resultado_compresion(res)
        render_pasos(res.pasos)
        render_arbol_huffman(res)

def tab_video() -> None:
    st.info("Visualización de conceptos H.264 integrada en el código fuente.")

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
