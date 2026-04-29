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

    The tree construction and code table are fully real.
    Bit-packing (``_comprimir_bytes``) is a stub that returns
    the *theoretical* size; replace it to inject real I/O.
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

    def _comprimir_bytes(self, codigos: Dict[str, str]) -> bytes:
        """
        ╔══════════════════════════════════════════════════════╗
        ║  STUB — replace this method with real bit-packing.   ║
        ║  Expected implementation:                            ║
        ║    1. Concatenate code strings into a bit-string.    ║
        ║    2. Pack every 8 bits into one output byte.        ║
        ║    3. Prepend the code table for self-contained file.║
        ╚══════════════════════════════════════════════════════╝
        Returns bytes of *theoretical* compressed length.
        """
        try:
            texto = self._datos.decode("utf-8", errors="replace")
        except Exception:
            texto = "".join(chr(b) for b in self._datos[:10_000])

        total_bits = sum(
            len(codigos.get(c, "?")) for c in texto
        )
        theo_bytes = math.ceil(total_bits / 8)
        return bytes(theo_bytes)  # placeholder payload

    def comprimir(self) -> ResultadoCompresion:
        t0 = time.perf_counter()

        try:
            texto = self._datos.decode("utf-8", errors="replace")
        except Exception:
            texto = self._datos.decode("latin-1", errors="replace")

        freq = Counter(texto)
        pasos: List[Dict[str, Any]] = [
            {
                "titulo": "Paso 1 · Análisis de Frecuencias",
                "detalle": (
                    f"Se cuentan las ocurrencias de cada símbolo en el texto.\n"
                    f"Total caracteres: {len(texto):,}  |  "
                    f"Símbolos únicos: {len(freq)}"
                ),
                "tabla": [
                    {"Símbolo": repr(s), "Frecuencia": f, "p(s)": round(f / len(texto), 6)}
                    for s, f in sorted(freq.items(), key=lambda x: -x[1])[:15]
                ],
            }
        ]

        raiz = self._construir_arbol(dict(freq))
        pasos.append(
            {
                "titulo": "Paso 2 · Construcción del Árbol Huffman",
                "detalle": (
                    "Algoritmo Greedy: se extraen los 2 nodos de menor frecuencia "
                    "del min-heap, se fusionan en un nodo padre y se reinsertan.\n"
                    f"Iteraciones de fusión: {len(freq) - 1}  |  "
                    f"Profundidad estimada del árbol: {math.ceil(math.log2(len(freq) + 1))}"
                ),
                "tabla": None,
            }
        )

        codigos = self._generar_codigos(raiz)
        tabla_codigos_muestra = [
            {
                "Símbolo": repr(s),
                "Código Huffman": c,
                "Longitud (bits)": len(c),
                "Freq": freq[s],
            }
            for s, c in sorted(codigos.items(), key=lambda x: len(x[1]))[:20]
        ]
        pasos.append(
            {
                "titulo": "Paso 3 · Generación de Códigos Binarios",
                "detalle": (
                    "Recorrido DFS del árbol: rama izquierda → '0', "
                    "rama derecha → '1'.\n"
                    "Propiedad preffix-free: ningún código es prefijo de otro."
                ),
                "tabla": tabla_codigos_muestra,
            }
        )

        comprimido = self._comprimir_bytes(codigos)
        L_bar = sum(
            (freq[s] / len(texto)) * len(codigos[s]) for s in codigos
        )
        H = -sum(
            (f / len(texto)) * math.log2(f / len(texto))
            for f in freq.values()
            if f > 0
        )
        pasos.append(
            {
                "titulo": "Paso 4 · Empaquetado de Bits (STUB)",
                "detalle": (
                    f"Cadena de bits total: {len(comprimido) * 8:,} bits\n"
                    f"Tamaño teórico comprimido: {len(comprimido):,} bytes\n"
                    f"L̄ (longitud promedio): {L_bar:.4f} bits/símbolo\n"
                    f"H(X) (entropía): {H:.4f} bits/símbolo\n"
                    ">>> Inyecta tu backend real aquí (bit-packing + tabla)."
                ),
                "tabla": None,
            }
        )

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
            es_stub=True,
        )


# ─── B. LZW ──────────────────────────────────────────────────────────────────

class CodificadorLZW:
    """
    Lempel–Ziv–Welch dictionary compression.

    Both compression and decompression are fully implemented.
    The output bytes pack each code into a 2-byte big-endian integer.
    """

    DICT_MAX = 4096  # LZW variant cap (12-bit codes)

    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        t0 = time.perf_counter()

        try:
            texto = datos.decode("utf-8", errors="replace")
        except Exception:
            texto = datos.decode("latin-1", errors="replace")

        texto = texto[:10_000]  # cap for UI performance

        diccionario: Dict[str, int] = {chr(i): i for i in range(256)}
        siguiente = 256
        w = ""
        codigos_salida: List[int] = []
        log_entradas: List[Dict] = []

        pasos: List[Dict[str, Any]] = [
            {
                "titulo": "Paso 1 · Inicialización del Diccionario",
                "detalle": (
                    "Se crean 256 entradas iniciales (ASCII 0–255).\n"
                    "El receptor construirá el mismo diccionario sin transmitirlo."
                ),
                "tabla": None,
            }
        ]

        for i, c in enumerate(texto):
            wc = w + c
            if wc in diccionario:
                w = wc
            else:
                codigos_salida.append(diccionario[w])
                if len(log_entradas) < 20:
                    log_entradas.append(
                        {
                            "Pos": i,
                            "Buffer w": repr(w),
                            "Char c": repr(c),
                            "Código emitido": diccionario[w],
                            "Nueva entrada": f"[{siguiente}] = {repr(wc)}",
                        }
                    )
                if siguiente < self.DICT_MAX:
                    diccionario[wc] = siguiente
                    siguiente += 1
                w = c

        if w:
            codigos_salida.append(diccionario[w])

        pasos.append(
            {
                "titulo": "Paso 2 · Búsqueda en Diccionario (Matching)",
                "detalle": (
                    f"Se extiende el buffer w mientras wc esté en el diccionario.\n"
                    f"Al no encontrar wc: se emite código de w y se agrega wc.\n"
                    f"Códigos emitidos: {len(codigos_salida):,}  |  "
                    f"Diccionario final: {siguiente} entradas"
                ),
                "tabla": log_entradas,
            }
        )

        # Pack as 2-byte big-endian codes
        if codigos_salida:
            comprimido = struct.pack(
                f">{len(codigos_salida)}H",
                *[min(c, 65535) for c in codigos_salida],
            )
        else:
            comprimido = b""

        pasos.append(
            {
                "titulo": "Paso 3 · Empaquetado de Códigos",
                "detalle": (
                    f"Cada código LZW se empaqueta en 2 bytes (big-endian).\n"
                    f"Bytes originales: {len(datos):,}  |  "
                    f"Bytes comprimidos: {len(comprimido):,}"
                ),
                "tabla": None,
            }
        )

        t1 = time.perf_counter()
        orig = len(datos)
        comp = max(1, len(comprimido))

        return ResultadoCompresion(
            nombre_algoritmo="LZW",
            datos_originales=datos,
            datos_comprimidos=comprimido,
            tamaño_original=orig,
            tamaño_comprimido=comp,
            tasa_compresion=orig / comp,
            ratio_reduccion=1 - (comp / orig),
            tiempo_ms=(t1 - t0) * 1000,
            pasos=pasos,
            tabla_lzw=log_entradas,
            es_stub=False,
        )


# ─── C. RLE ───────────────────────────────────────────────────────────────────

class CodificadorRLE:
    """Run-Length Encoding — fully implemented, lossless."""

    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        t0 = time.perf_counter()

        if not datos:
            return ResultadoCompresion(
                "RLE", b"", b"", 0, 0, 1.0, 0.0, 0.0
            )

        comprimido = bytearray()
        runs_log: List[Dict] = []
        i = 0

        while i < len(datos):
            byte_actual = datos[i]
            conteo = 1
            while (
                i + conteo < len(datos)
                and datos[i + conteo] == byte_actual
                and conteo < 255
            ):
                conteo += 1

            comprimido.extend([conteo, byte_actual])
            if len(runs_log) < 25:
                runs_log.append(
                    {
                        "Posición": i,
                        "Byte": f"0x{byte_actual:02X}",
                        "Run": conteo,
                        "Salida": f"[{conteo}, 0x{byte_actual:02X}]",
                        "¿Eficiente?": "✓" if conteo > 2 else "✗ overhead",
                    }
                )
            i += conteo

        pasos: List[Dict[str, Any]] = [
            {
                "titulo": "Paso 1 · Escaneo de Secuencias",
                "detalle": (
                    "Se recorre el byte array linealmente.\n"
                    "Para cada posición se cuenta cuántos bytes consecutivos "
                    "son iguales (run).\n"
                    f"Total runs encontrados: {len(runs_log):,}"
                ),
                "tabla": runs_log,
            },
            {
                "titulo": "Paso 2 · Codificación [conteo, valor]",
                "detalle": (
                    "Cada run se codifica como el par (conteo, byte).\n"
                    "Formato: [N, X] → N repeticiones del byte X.\n"
                    f"Bytes originales: {len(datos):,}  |  "
                    f"Bytes comprimidos: {len(comprimido):,}\n"
                    f"Overhead si todos los runs tienen N=1: 2× el tamaño original."
                ),
                "tabla": None,
            },
        ]

        t1 = time.perf_counter()
        orig = len(datos)
        comp = max(1, len(comprimido))

        return ResultadoCompresion(
            nombre_algoritmo="RLE",
            datos_originales=datos,
            datos_comprimidos=bytes(comprimido),
            tamaño_original=orig,
            tamaño_comprimido=comp,
            tasa_compresion=orig / comp,
            ratio_reduccion=1 - (comp / orig),
            tiempo_ms=(t1 - t0) * 1000,
            pasos=pasos,
            es_stub=False,
        )


# ─── D. DCT (Image / JPEG-like) ───────────────────────────────────────────────

class CodificadorDCT:
    """
    Discrete Cosine Transform — simplified JPEG-pipeline.

    Quantization and entropy stages are stubs; replace to
    implement a complete JPEG encoder.
    """

    # Standard JPEG luminance quantization table
    Q_LUMA = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ],
        dtype=float,
    )

    def _dct2d_manual(self, bloque: np.ndarray) -> np.ndarray:
        """DCT-II 2D on an 8×8 block using the definition formula."""
        N = 8
        b = bloque.astype(float) - 128.0
        F = np.zeros((N, N))
        for u in range(N):
            for v in range(N):
                cu = 1.0 / math.sqrt(2) if u == 0 else 1.0
                cv = 1.0 / math.sqrt(2) if v == 0 else 1.0
                xs = np.arange(N)
                ys = np.arange(N)
                cos_u = np.cos((2 * xs + 1) * u * math.pi / (2 * N))
                cos_v = np.cos((2 * ys + 1) * v * math.pi / (2 * N))
                F[u, v] = (2 / N) * cu * cv * np.sum(
                    b * cos_u[:, None] * cos_v[None, :]
                )
        return F

    def comprimir(
        self, datos: bytes, calidad: int = 50
    ) -> ResultadoCompresion:
        """
        ╔══════════════════════════════════════════════════════╗
        ║  PARTIAL STUB — DCT is real; quantization is real.  ║
        ║  ZigZag scan + entropy coding (Huffman) → STUB.     ║
        ╚══════════════════════════════════════════════════════╝
        """
        t0 = time.perf_counter()

        # Pad/extract a sample 8×8 block from the data
        raw = list(datos[:64]) + [128] * max(0, 64 - len(datos))
        bloque = np.array(raw[:64], dtype=float).reshape(8, 8)

        # Real DCT
        F = self._dct2d_manual(bloque)
        dc_coef = F[0, 0]
        energia_dc = float(F[0, 0] ** 2)
        energia_total = float(np.sum(F**2))
        energia_pct = 100 * energia_dc / energia_total if energia_total > 0 else 0

        # Real quantization
        scale = max(1, (100 - calidad) / 50) if calidad < 50 else 50 / max(1, calidad)
        Q = self.Q_LUMA * scale
        F_q = np.round(F / Q).astype(int)
        ceros_ac = int(np.sum(F_q[1:] == 0))

        pasos = [
            {
                "titulo": "Paso 1 · Partición en Bloques 8×8",
                "detalle": (
                    "La imagen se divide en bloques de 8×8 píxeles.\n"
                    "Cada bloque se procesa independientemente.\n"
                    "Level shift: se resta 128 a cada valor (rango [0,255] → [-128,127])."
                ),
                "tabla": None,
            },
            {
                "titulo": "Paso 2 · DCT-II 2D",
                "detalle": (
                    "F(u,v) = (2/N)·Cᵤ·Cᵥ·ΣΣ f(x,y)·cos[(2x+1)uπ/2N]·cos[(2y+1)vπ/2N]\n\n"
                    f"Coeficiente DC (energía): {dc_coef:.2f}\n"
                    f"Concentración de energía en DC: {energia_pct:.1f}% del total\n"
                    "→ Esto permite descartar componentes de alta frecuencia."
                ),
                "tabla": [
                    {"u": u, "v": v, "F(u,v)": round(float(F[u, v]), 3)}
                    for u in range(4)
                    for v in range(4)
                ],
            },
            {
                "titulo": "Paso 3 · Cuantización (tabla JPEG)",
                "detalle": (
                    f"F_q(u,v) = round(F(u,v) / Q(u,v))\n"
                    f"Factor de calidad Q={calidad}  →  scale={scale:.2f}\n"
                    f"Coeficientes AC puestos a 0: {ceros_ac}/63  "
                    f"({100*ceros_ac/63:.0f}% descartados)\n"
                    "→ Mayor Q = más ceros AC = mejor compresión (con pérdida)."
                ),
                "tabla": [
                    {"u": u, "v": v, "F(u,v)": round(float(F[u, v]), 2), "Q(u,v)": round(float(Q[u, v]), 2), "F_q": int(F_q[u, v])}
                    for u in range(4)
                    for v in range(4)
                ],
            },
            {
                "titulo": "Paso 4 · Escaneo ZigZag + Codificación Entrópica (STUB)",
                "detalle": (
                    "ZigZag reordena la matriz 8×8 → vector 1D para agrupar ceros.\n"
                    "Run-Length de los ceros AC + Huffman sobre pares (run, magnitud).\n"
                    ">>> Inyecta tu implementación de ZigZag + Huffman AC/DC aquí."
                ),
                "tabla": None,
            },
        ]

        orig = len(datos)
        theo_ratio = max(0.05, 1 - (calidad / 100) * 0.92)
        comp = max(1, int(orig * theo_ratio))
        t1 = time.perf_counter()

        return ResultadoCompresion(
            nombre_algoritmo="DCT — JPEG-like",
            datos_originales=datos,
            datos_comprimidos=bytes(comp),
            tamaño_original=orig,
            tamaño_comprimido=comp,
            tasa_compresion=orig / comp,
            ratio_reduccion=1 - (comp / orig),
            tiempo_ms=(t1 - t0) * 1000,
            pasos=pasos,
            es_stub=True,
        )


# ─── E. μ-Law (Audio) ────────────────────────────────────────────────────────

class CodificadorMuLaw:
    """
    G.711 μ-law companding — fully implemented.
    Maps 16-bit linear PCM → 8-bit logarithmic samples (2:1 ratio).
    """

    MU = 255
    BIAS = 132

    def _encode_sample(self, sample: int) -> int:
        """G.711 μ-law encoding for a single 16-bit PCM sample."""
        sample = max(-32768, min(32767, sample))
        sign = 0x00 if sample >= 0 else 0x80
        magnitude = min(abs(sample), 32635) + self.BIAS
        exp = max(0, min(int(math.log2(magnitude)) - 7, 7))
        mantissa = (magnitude >> (exp + 3)) & 0x0F
        return (~(sign | (exp << 4) | mantissa)) & 0xFF

    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        t0 = time.perf_counter()

        n_samples = len(datos) // 2
        samples: Tuple[int, ...] = struct.unpack(
            f"<{n_samples}h", datos[: n_samples * 2]
        )

        encoded = bytes(self._encode_sample(s) for s in samples)

        muestra_log = [
            {
                "Muestra PCM (16-bit)": s,
                "μ-Law (8-bit)": self._encode_sample(s),
                "Representación hex": f"0x{self._encode_sample(s):02X}",
            }
            for s in list(samples[:15])
        ]

        pasos = [
            {
                "titulo": "Paso 1 · Lectura de Muestras PCM 16-bit",
                "detalle": (
                    f"Se interpretan los bytes de audio como muestras signed 16-bit little-endian.\n"
                    f"Total muestras leídas: {n_samples:,}  |  "
                    f"Rango: [{min(samples)}, {max(samples)}]"
                ),
                "tabla": muestra_log,
            },
            {
                "titulo": "Paso 2 · Companding Logarítmico μ-Law",
                "detalle": (
                    f"y = sgn(x)·ln(1 + μ|x|) / ln(1 + μ)  donde μ = {self.MU}\n"
                    "Codificación digital G.711:\n"
                    "  sign   = bit 7\n"
                    "  exp    = bits 6-4 (exponente de magnitud)\n"
                    "  mantissa = bits 3-0\n"
                    "Inversión del resultado (complemento de 1)."
                ),
                "tabla": None,
            },
            {
                "titulo": "Paso 3 · Resultado",
                "detalle": (
                    f"Muestras 16-bit: {n_samples:,}  →  Bytes 8-bit: {len(encoded):,}\n"
                    f"Reducción exacta: 2:1 (1 byte por muestra vs 2 bytes)\n"
                    f"Tiempo: {(time.perf_counter() - t0)*1000:.2f} ms"
                ),
                "tabla": None,
            },
        ]

        t1 = time.perf_counter()
        orig = len(datos)
        comp = max(1, len(encoded))

        return ResultadoCompresion(
            nombre_algoritmo="μ-Law G.711",
            datos_originales=datos,
            datos_comprimidos=encoded,
            tamaño_original=orig,
            tamaño_comprimido=comp,
            tasa_compresion=orig / comp,
            ratio_reduccion=1 - (comp / orig),
            tiempo_ms=(t1 - t0) * 1000,
            pasos=pasos,
            es_stub=False,
        )


# ─── F. ADPCM (Audio — Stub) ─────────────────────────────────────────────────

class CodificadorADPCM:
    """
    Adaptive Differential Pulse-Code Modulation.

    ╔══════════════════════════════════════════════════════╗
    ║  STUB: step_size adaptation is placeholder.          ║
    ║  Replace _compute_delta() with IMA-ADPCM tables.    ║
    ╚══════════════════════════════════════════════════════╝
    """

    def comprimir(self, datos: bytes) -> ResultadoCompresion:
        t0 = time.perf_counter()

        n_samples = len(datos) // 2
        samples: Tuple[int, ...] = struct.unpack(
            f"<{n_samples}h", datos[: n_samples * 2]
        )

        # --- STUB: naive delta coding (not real ADPCM) ---
        predictor = 0
        step_size = 16  # fixed — real ADPCM adapts this
        deltas: List[int] = []
        for s in samples[:20]:
            delta = (s - predictor) // step_size
            delta = max(-8, min(7, delta))  # 4-bit range
            deltas.append(delta)
            predictor += delta * step_size  # update predictor

        pasos = [
            {
                "titulo": "Paso 1 · Predicción (STUB)",
                "detalle": (
                    "xₙ_pred = xₙ₋₁  (predictor de orden 1)\n"
                    "Δₙ = (xₙ − xₙ_pred) / step_size\n"
                    f"Se cuantiza Δₙ a 4 bits: rango [-8, 7]\n"
                    ">>> STUB: usar tabla IMA-ADPCM para step_size adaptivo."
                ),
                "tabla": [
                    {"Muestra (16-bit)": s, "Delta 4-bit": d}
                    for s, d in zip(list(samples[:20]), deltas)
                ],
            },
            {
                "titulo": "Paso 2 · Empaquetado 4-bit (STUB)",
                "detalle": (
                    "Dos deltas de 4 bits se empaquetan en 1 byte.\n"
                    f"Teórico: {n_samples:,} muestras → {n_samples // 2:,} bytes (4:1 ratio).\n"
                    ">>> Inyecta empaquetado real con IMA-ADPCM aquí."
                ),
                "tabla": None,
            },
            {
                "titulo": "Paso 3 · Adaptación del Step Size (STUB)",
                "detalle": (
                    "El step_size se incrementa cuando el delta es grande "
                    "(señal rápida) y decrece cuando es pequeño (señal lenta).\n"
                    ">>> Usar la tabla de índice IMA-ADPCM para la adaptación."
                ),
                "tabla": None,
            },
        ]

        orig = len(datos)
        comp = max(1, orig // 4)
        t1 = time.perf_counter()

        return ResultadoCompresion(
            nombre_algoritmo="ADPCM (IMA)",
            datos_originales=datos,
            datos_comprimidos=bytes(comp),
            tamaño_original=orig,
            tamaño_comprimido=comp,
            tasa_compresion=orig / comp,
            ratio_reduccion=1 - (comp / orig),
            tiempo_ms=(t1 - t0) * 1000,
            pasos=pasos,
            es_stub=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  SOURCE CODE STRINGS  (shown in st.code() blocks)
# ═════════════════════════════════════════════════════════════════════════════

_CODE_HUFFMAN = """\
# ╔══════════════════════════════════════════════════════════╗
# ║  HUFFMAN CODING — Fundamento Matemático Completo         ║
# ╚══════════════════════════════════════════════════════════╝
import heapq
from collections import Counter
import math

# ── 1. Frecuencias y probabilidades ──────────────────────────
def analizar(texto: str) -> dict:
    freq = Counter(texto)
    N = len(texto)
    H = -sum((f/N) * math.log2(f/N) for f in freq.values())
    print(f"Entropía H(X) = {H:.4f} bits/símbolo")
    return freq

# ── 2. Árbol de Huffman ───────────────────────────────────────
class Nodo:
    def __init__(self, sym, freq):
        self.sym, self.freq = sym, freq
        self.izq = self.der = None
    def __lt__(self, o): return self.freq < o.freq   # para heapq

def construir_arbol(freq: dict) -> Nodo:
    heap = [Nodo(s, f) for s, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        L = heapq.heappop(heap)   # nodo de menor frecuencia
        R = heapq.heappop(heap)   # segundo menor
        padre = Nodo(None, L.freq + R.freq)
        padre.izq, padre.der = L, R
        heapq.heappush(heap, padre)
    return heap[0]  # raíz

# ── 3. Generación de códigos (DFS) ────────────────────────────
def generar_codigos(nodo: Nodo, prefijo="", codigos=None) -> dict:
    if codigos is None: codigos = {}
    if nodo.sym is not None:
        codigos[nodo.sym] = prefijo or "0"   # hoja
    else:
        generar_codigos(nodo.izq, prefijo + "0", codigos)
        generar_codigos(nodo.der, prefijo + "1", codigos)
    return codigos

# ── 4. Codificación a bits ────────────────────────────────────
def codificar(texto: str, codigos: dict) -> bytes:
    bits = "".join(codigos[c] for c in texto)
    # Empaquetado: cada 8 bits → 1 byte
    n_bytes = math.ceil(len(bits) / 8)
    bits_padded = bits.ljust(n_bytes * 8, '0')
    return bytes(
        int(bits_padded[i:i+8], 2) for i in range(0, len(bits_padded), 8)
    )

# ── 5. Métricas ───────────────────────────────────────────────
def metricas(texto, codigos):
    N = len(texto)
    freq = Counter(texto)
    L_bar = sum((freq[s]/N) * len(codigos[s]) for s in codigos)
    H = -sum((f/N)*math.log2(f/N) for f in freq.values())
    eta = H / L_bar
    print(f"L̄ = {L_bar:.4f} bits/símbolo")
    print(f"η = {eta*100:.2f}%   R = {(1-eta)*100:.2f}%")
    print(f"Garantía Huffman: L̄ ≤ H + 1 → {L_bar:.4f} ≤ {H+1:.4f}")
"""

_CODE_LZW = """\
# ╔══════════════════════════════════════════════════════════╗
# ║  LZW — Lempel-Ziv-Welch                                  ║
# ╚══════════════════════════════════════════════════════════╝

# ── Compresión ────────────────────────────────────────────────
def lzw_comprimir(texto: str) -> list[int]:
    # Diccionario inicial: 256 entradas ASCII
    dic = {chr(i): i for i in range(256)}
    sig = 256
    w, salida = "", []

    for c in texto:
        wc = w + c
        if wc in dic:
            w = wc                     # extender buffer
        else:
            salida.append(dic[w])      # emitir código de w
            dic[wc] = sig              # nueva entrada
            sig += 1
            w = c                      # reset buffer

    if w: salida.append(dic[w])
    return salida                      # lista de enteros

# ── Descompresión ─────────────────────────────────────────────
def lzw_descomprimir(codigos: list[int]) -> str:
    dic = {i: chr(i) for i in range(256)}
    sig = 256
    w = chr(codigos[0])
    resultado = [w]

    for codigo in codigos[1:]:
        if codigo in dic:
            entrada = dic[codigo]
        elif codigo == sig:
            entrada = w + w[0]         # caso especial LZW
        else:
            raise ValueError(f"Código inválido: {codigo}")
        resultado.append(entrada)
        dic[sig] = w + entrada[0]     # reconstruye el diccionario
        sig += 1
        w = entrada

    return "".join(resultado)

# ── Análisis de compresión ────────────────────────────────────
def analizar_lzw(texto: str):
    codigos = lzw_comprimir(texto)
    bits_orig = len(texto) * 8
    bits_comp = len(codigos) * 12     # 12 bits por código LZW
    ratio = bits_orig / bits_comp
    print(f"Códigos: {len(codigos)}  |  Ratio: {ratio:.2f}:1")
    print(f"Reducción: {(1 - 1/ratio)*100:.1f}%")
"""

_CODE_RLE = """\
# ╔══════════════════════════════════════════════════════════╗
# ║  RLE — Run-Length Encoding                               ║
# ╚══════════════════════════════════════════════════════════╝

# ── Compresión ────────────────────────────────────────────────
def rle_comprimir(datos: bytes) -> bytes:
    \"\"\"
    Complejidad: O(n)
    Mejor caso: n bytes iguales → 2 bytes de salida
    Peor caso:  datos aleatorios → 2n bytes (overhead 2×)
    \"\"\"
    out, i = bytearray(), 0
    while i < len(datos):
        byte = datos[i]
        run = 1
        while i + run < len(datos) and datos[i+run] == byte and run < 255:
            run += 1
        out.extend([run, byte])
        i += run
    return bytes(out)

# ── Descompresión ─────────────────────────────────────────────
def rle_descomprimir(datos: bytes) -> bytes:
    out, i = bytearray(), 0
    while i + 1 < len(datos):
        out.extend([datos[i+1]] * datos[i])
        i += 2
    return bytes(out)

# ── RLE para imágenes (variante BMP) ─────────────────────────
def rle_bmp(pixels: list[int]) -> list[tuple]:
    \"\"\"Codificación RLE típica de BMP (modo 8-bit).\"\"\"
    runs, i = [], 0
    while i < len(pixels):
        val, n = pixels[i], 1
        while i+n < len(pixels) and pixels[i+n] == val and n < 255:
            n += 1
        if n >= 3:
            runs.append(('RUN', n, val))   # (n, val) para n≥3
        else:
            runs.append(('LIT', n, val))   # literal para n<3
        i += n
    return runs
"""

_CODE_DCT = """\
# ╔══════════════════════════════════════════════════════════╗
# ║  DCT — Transformada Discreta del Coseno (JPEG pipeline)  ║
# ╚══════════════════════════════════════════════════════════╝
import numpy as np, math

# ── DCT-II 2D (scipy-style) ───────────────────────────────────
def dct2(bloque: np.ndarray) -> np.ndarray:
    \"\"\"
    F(u,v) = (2/N)·Cᵤ·Cᵥ·ΣΣ f(x,y)·cos[(2x+1)uπ/2N]·cos[(2y+1)vπ/2N]
    \"\"\"
    N, b = bloque.shape[0], bloque.astype(float) - 128
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

# ── Tabla de cuantización JPEG (luminancia) ───────────────────
Q = np.array([
    [16,11,10,16,24,40,51,61], [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56], [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99],
])

# ── Pipeline JPEG simplificado ────────────────────────────────
def jpeg_pipeline(img_gray: np.ndarray, quality: int = 50):
    scale = max(1,(100-quality)/50) if quality < 50 else 50/max(1,quality)
    Q_scaled = Q * scale

    for i in range(0, img_gray.shape[0]-7, 8):
        for j in range(0, img_gray.shape[1]-7, 8):
            bloque = img_gray[i:i+8, j:j+8]
            F = dct2(bloque)            # 1. DCT
            F_q = np.round(F/Q_scaled)  # 2. Cuantización
            # 3. ZigZag → vector 1D (STUB)
            # 4. RLE de ceros AC  (STUB)
            # 5. Huffman DC + AC  (STUB)
"""

_CODE_MULAW = """\
# ╔══════════════════════════════════════════════════════════╗
# ║  μ-Law Companding — G.711 (Audio Telefónico)             ║
# ╚══════════════════════════════════════════════════════════╝
import math, struct

MU, BIAS = 255, 132

def encode_mulaw(sample: int) -> int:
    \"\"\"
    Compresión logarítmica: 16-bit PCM → 8-bit μ-law.
    y = sgn(x)·ln(1 + μ|x|) / ln(1 + μ)
    \"\"\"
    sample  = max(-32768, min(32767, sample))
    sign    = 0x00 if sample >= 0 else 0x80
    mag     = min(abs(sample), 32635) + BIAS
    exp     = max(0, min(int(math.log2(mag)) - 7, 7))
    mant    = (mag >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mant)) & 0xFF

def decode_mulaw(byte: int) -> int:
    \"\"\"Decodificación μ-law: 8-bit → 16-bit PCM.\"\"\"
    byte = ~byte & 0xFF
    sign = byte & 0x80
    exp  = (byte >> 4) & 0x07
    mant = byte & 0x0F
    mag  = ((mant << 1) | 1) << (exp + 2)
    return -(mag - BIAS) if sign else (mag - BIAS)

def comprimir_audio(pcm_bytes: bytes) -> bytes:
    n  = len(pcm_bytes) // 2
    muestras = struct.unpack(f'<{n}h', pcm_bytes[:n*2])
    return bytes(encode_mulaw(s) for s in muestras)
"""

_CODE_ADPCM = """\
# ╔══════════════════════════════════════════════════════════╗
# ║  IMA-ADPCM — Adaptive Differential PCM                   ║
# ╚══════════════════════════════════════════════════════════╝

# Tabla de pasos IMA-ADPCM (88 niveles)
STEP_TABLE = [
    7,8,9,10,11,12,13,14,16,17,19,21,23,25,28,31,34,37,41,45,
    50,55,60,66,73,80,88,97,107,118,130,143,157,173,190,209,230,
    253,279,307,337,371,408,449,494,544,598,658,724,796,876,963,
    1060,1166,1282,1411,1552,1707,1878,2066,2272,2499,2749,3024,
    3327,3660,4026,4428,4871,5358,5894,6484,7132,7845,8630,9493,
    10442,11487,12635,13899,15289,16818,18500,20350,22385,24623,27086,
    29794,32767
]
INDEX_TABLE = [-1,-1,-1,-1,2,4,6,8,-1,-1,-1,-1,2,4,6,8]

def ima_adpcm_encode(pcm_bytes: bytes) -> bytes:
    \"\"\"
    IMA-ADPCM: 16-bit PCM → 4-bit delta codes → empaquetado 2 por byte.
    Tasa de compresión: 4:1
    \"\"\"
    import struct
    n = len(pcm_bytes) // 2
    samples = struct.unpack(f'<{n}h', pcm_bytes[:n*2])

    predictor, index = 0, 0
    output = bytearray()
    nibbles = []

    for sample in samples:
        step = STEP_TABLE[index]
        diff = sample - predictor
        nibble = 0
        if diff < 0:
            nibble, diff = 8, -diff

        # Cuantizar la diferencia a 3 bits de magnitud
        if diff >= step:       nibble |= 4; diff -= step
        step >>= 1
        if diff >= step:       nibble |= 2; diff -= step
        step >>= 1
        if diff >= step:       nibble |= 1

        # Actualizar predictor y step size
        step = STEP_TABLE[index]
        delta = step >> 3
        if nibble & 4: delta += step
        if nibble & 2: delta += step >> 1
        if nibble & 1: delta += step >> 2
        predictor += delta if not (nibble & 8) else -delta
        predictor  = max(-32768, min(32767, predictor))
        index      = max(0, min(88, index + INDEX_TABLE[nibble & 7]))
        nibbles.append(nibble & 0xF)

    # Empaquetar 2 nibbles por byte
    for i in range(0, len(nibbles)-1, 2):
        output.append(nibbles[i] | (nibbles[i+1] << 4))
    return bytes(output)
"""

_CODE_ENTROPIA = """\
# ╔══════════════════════════════════════════════════════════╗
# ║  FUNDAMENTO MATEMÁTICO — Teoría de la Información        ║
# ╚══════════════════════════════════════════════════════════╝
import math
from collections import Counter

def entropia_shannon(datos: str | bytes) -> float:
    \"\"\"
    H(X) = −∑ p(xᵢ)·log₂ p(xᵢ)   [bits/símbolo]

    Casos límite:
      H = 0      → todos los símbolos iguales (máx. compresible)
      H = log₂N  → distribución uniforme (máx. entropía)
    \"\"\"
    freq = Counter(datos)
    N = sum(freq.values())
    return -sum((f/N)*math.log2(f/N) for f in freq.values() if f > 0)

def longitud_promedio(probs: dict, codigos: dict) -> float:
    \"\"\"L̄ = ∑ p(xᵢ)·l(xᵢ)  donde l(xᵢ) = longitud del código\"\"\"
    return sum(probs[s]*len(codigos[s]) for s in probs if s in codigos)

def eficiencia(H: float, L_bar: float) -> float:
    \"\"\"η = H(X) / L̄   (idealmente η → 1)\"\"\"
    return H / L_bar if L_bar > 0 else 0.0

def redundancia(eta: float) -> float:
    \"\"\"R = 1 − η   (fracción de bits redundantes)\"\"\"
    return 1.0 - eta

def capacidad_canal_shannon(S_N_dB: float, B_Hz: float) -> float:
    \"\"\"
    Teorema de Shannon-Hartley:
    C = B · log₂(1 + S/N)   [bits/segundo]
    \"\"\"
    S_N_lineal = 10 ** (S_N_dB / 10)
    return B_Hz * math.log2(1 + S_N_lineal)
"""


# ═════════════════════════════════════════════════════════════════════════════
#  UI HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def fmt_bytes(n: int) -> str:
    """Human-readable byte size."""
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n/1024:.2f} KB"
    return f"{n/1024**2:.2f} MB"


def render_header() -> None:
    """App header with badges."""
    st.markdown(
        """
        <div class="app-header-wrap">
          <div class="app-header">
            <div class="header-left">
              <div class="header-icon">⚡</div>
              <div>
                <p class="app-title">DataLab · Compresión</p>
                <p class="app-subtitle">
                  Shannon Entropy · Huffman · LZW · RLE · DCT · μ-Law · ADPCM
                </p>
              </div>
            </div>
            <div class="header-badges">
              <span class="hbadge hbadge-c">Texto</span>
              <span class="hbadge hbadge-v">Imagen</span>
              <span class="hbadge hbadge-g">Audio</span>
              <span class="hbadge hbadge-c">Video</span>
              <span class="hbadge hbadge-v">v1.0.0</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(icon: str, text: str) -> None:
    st.markdown(
        f"""<div class="section-label">
              <span class="sl-icon">{icon}</span>
              <span class="sl-text">{text}</span>
            </div>""",
        unsafe_allow_html=True,
    )


def info_box(icon: str, content: str, variant: str = "") -> None:
    st.markdown(
        f"""<div class="info-box {variant}">
              <div class="ib-icon">{icon}</div>
              <div class="ib-content">{content}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def formula_display(formula: str) -> None:
    st.markdown(
        f'<div class="formula-display">{formula}</div>',
        unsafe_allow_html=True,
    )


def render_dashboard(stats: EstadisticasInfo) -> None:
    """Four-column metric dashboard."""
    c1, c2, c3, c4 = st.columns(4)
    eta_pct = stats.eficiencia * 100
    with c1:
        st.metric(
            "Entropía H(X)",
            f"{stats.entropia:.4f}",
            "bits / símbolo",
        )
    with c2:
        st.metric(
            "Long. Promedio L̄",
            f"{stats.longitud_promedio:.4f}",
            "bits / símbolo",
        )
    with c3:
        st.metric(
            "Eficiencia η",
            f"{eta_pct:.2f}%",
            f"Redundancia {stats.redundancia*100:.2f}%",
        )
    with c4:
        st.metric(
            "Alfabeto",
            f"{stats.simbolos_unicos}",
            f"de {stats.total_simbolos:,} símbolos",
        )

    # Entropy bar (relative to max)
    pct = stats.entropia / stats.entropia_maxima if stats.entropia_maxima > 0 else 0
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:0.75rem;margin-top:0.4rem">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                        color:var(--muted);white-space:nowrap">
            H vs H_max
          </span>
          <div class="entropy-bar-bg" style="flex:1">
            <div class="entropy-bar-fill" style="width:{pct*100:.1f}%"></div>
          </div>
          <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                        color:var(--cyan);white-space:nowrap">
            {pct*100:.1f}%&nbsp;de&nbsp;{stats.entropia_maxima:.3f} bits
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tabla_simbolos(stats: EstadisticasInfo, max_rows: int = 35) -> None:
    """Styled symbol frequency/probability table."""
    rows = []
    for sym, freq in sorted(
        stats.frecuencias.items(), key=lambda x: -x[1]
    )[:max_rows]:
        prob = stats.probabilidades[sym]
        info_bits = -math.log2(prob) if prob > 0 else 0
        l_approx = math.ceil(info_bits) if info_bits > 0 else 1
        contrib = prob * info_bits
        rows.append(
            {
                "Símbolo": repr(sym) if isinstance(sym, str) else f"0x{sym:02X}",
                "Frecuencia": freq,
                "Probabilidad p(x)": round(prob, 8),
                "Información I(x) [bits]": round(info_bits, 4),
                "Long. cód. aprox. lᵢ": l_approx,
                "Contribución pᵢ·lᵢ": round(contrib, 6),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format(
            {
                "Probabilidad p(x)": "{:.6f}",
                "Información I(x) [bits]": "{:.4f}",
                "Contribución pᵢ·lᵢ": "{:.6f}",
            }
        ).background_gradient(subset=["Probabilidad p(x)"], cmap="YlOrBr"),
        use_container_width=True,
        height=380,
    )


def render_resultado_compresion(res: ResultadoCompresion) -> None:
    """Compression result dashboard."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Tamaño Original", fmt_bytes(res.tamaño_original))
    with c2:
        st.metric("Tamaño Comprimido", fmt_bytes(res.tamaño_comprimido))
    with c3:
        st.metric("Tasa", f"{res.tasa_compresion:.2f}:1")
    with c4:
        st.metric("Reducción", f"{max(0,res.ratio_reduccion)*100:.1f}%")

    # Visual bar
    ratio = max(0.0, min(1.0, res.ratio_reduccion))
    st.markdown(
        f"""
        <div style="margin:0.6rem 0 0.2rem">
          <div class="result-bar-wrap">
            <div class="result-bar-fill" style="width:{ratio*100:.1f}%"></div>
          </div>
          <div style="display:flex;justify-content:space-between;
                      font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                      color:var(--muted);margin-top:0.3rem">
            <span>0% reducción</span>
            <span style="color:var(--cyan)">
              {ratio*100:.1f}% reducido en {res.tiempo_ms:.1f} ms
            </span>
            <span>100% reducción</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if res.es_stub:
        info_box(
            "🔧",
            "<strong>Modo Educativo (STUB):</strong> Los bytes de salida son "
            "teóricos. La estadística y estructura del algoritmo son reales.<br>"
            "Reemplaza las funciones marcadas con <code>STUB</code> para "
            "obtener un compresor completo.",
            "amber",
        )


def render_pasos(pasos: List[Dict[str, Any]]) -> None:
    """Visual step-by-step procedure display."""
    section_label("🔍", "PROCEDIMIENTO PASO A PASO")
    for i, paso in enumerate(pasos, 1):
        titulo = paso.get("titulo", f"Paso {i}")
        detalle = paso.get("detalle", "")
        tabla = paso.get("tabla", None)
        with st.expander(titulo, expanded=(i <= 2)):
            if detalle:
                st.markdown(
                    f"<div style='font-family:\"IBM Plex Mono\",monospace;"
                    f"font-size:0.76rem;color:var(--txt-dim);line-height:1.8;"
                    f"white-space:pre-wrap'>{detalle}</div>",
                    unsafe_allow_html=True,
                )
            if tabla:
                st.dataframe(
                    pd.DataFrame(tabla), use_container_width=True, height=250
                )


def render_huffman_codes(codigos: Dict[str, str]) -> None:
    """Visual Huffman code table with bit-length bar."""
    section_label("🌳", "TABLA DE CÓDIGOS HUFFMAN")
    rows = [
        {
            "Símbolo": repr(s),
            "Código": c,
            "Longitud (bits)": len(c),
        }
        for s, c in sorted(codigos.items(), key=lambda x: len(x[1]))[:30]
    ]
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.background_gradient(subset=["Longitud (bits)"], cmap="Blues"),
        use_container_width=True,
        height=350,
    )
    # Tiny visual representation
    html_codes = "".join(
        f'<span class="huffman-code-row">'
        f'<span class="huffman-sym">{repr(s)}</span>'
        f'<span style="color:var(--muted)">→</span>'
        f'<span class="huffman-bits">{c}</span>'
        f'<span class="huffman-len">({len(c)}b)</span>'
        f"</span>"
        for s, c in sorted(codigos.items(), key=lambda x: len(x[1]))[:20]
    )
    st.markdown(
        f"<div style='margin-top:0.5rem;line-height:2.2'>{html_codes}</div>",
        unsafe_allow_html=True,
    )


def render_no_file(icon: str, texto: str, subtexto: str) -> None:
    st.markdown(
        f"""
        <div class="no-file-state">
          <div class="nfs-icon">{icon}</div>
          <div class="nfs-title">{texto}</div>
          <div class="nfs-sub">{subtexto}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_formulas_matematicas() -> None:
    """Expander with LaTeX formulas."""
    with st.expander("📐 Fundamento Matemático — Teoría de la Información", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Entropía de Shannon**")
            st.latex(r"H(X) = -\sum_{i=1}^{N} p(x_i) \cdot \log_2 p(x_i)")
            st.markdown("**Longitud Promedio**")
            st.latex(r"\bar{L} = \sum_{i=1}^{N} p(x_i) \cdot l_i")
        with col2:
            st.markdown("**Eficiencia y Redundancia**")
            st.latex(r"\eta = \frac{H(X)}{\bar{L}}, \quad R = 1 - \eta")
            st.markdown("**Cota de Huffman (Teorema de Shannon)**")
            st.latex(r"H(X) \leq \bar{L}_{Huffman} < H(X) + 1")
        st.markdown("**Capacidad de Canal (Shannon-Hartley)**")
        st.latex(r"C = B \cdot \log_2\!\left(1 + \frac{S}{N}\right) \text{ [bits/s]}")


# ═════════════════════════════════════════════════════════════════════════════
#  TAB IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════

# ─── TAB 1: TEXTO ────────────────────────────────────────────────────────────

def tab_texto() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")

    with col_up:
        section_label("📄", "CARGAR ARCHIVO DE TEXTO")
        uploaded = st.file_uploader(
            "Sube un archivo de texto",
            type=["txt", "csv", "json", "xml", "md", "log", "py", "html"],
            key="uploader_texto",
            label_visibility="collapsed",
        )

    with col_info:
        section_label("ℹ️", "ALGORITMOS DISPONIBLES")
        for algo_info in [
            ("Huffman", "Codificación óptima por símbolo.\nÁrbol binario mínimo de entropía.", "tag-c"),
            ("LZW", "Compresión por diccionario adaptivo.\nBase de GIF y PDF.", "tag-v"),
            ("RLE", "Codificación por longitud de carrera.\nÓptimo para datos repetitivos.", "tag-g"),
        ]:
            st.markdown(
                f"""<div class="algo-card">
                      <div class="ac-top">
                        <span class="ac-name">{algo_info[0]}</span>
                        <span class="ac-tag {algo_info[2]}">lossless</span>
                      </div>
                      <div class="ac-desc">{algo_info[1]}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    if uploaded is None:
        render_no_file(
            "📝",
            "Sube un archivo para comenzar el análisis",
            "Formatos soportados: .txt  .csv  .json  .xml  .md  .py  .log",
        )
        return

    datos = uploaded.read()
    analizador = AnalizadorTexto(datos, uploaded.name)

    with st.spinner("Calculando entropía y estadísticas…"):
        stats = analizador.calcular_todo()

    # ── Dashboard ─────────────────────────────────────────────
    section_label("📊", "DASHBOARD INFORMATIVO")
    st.markdown(
        f"""<div class="card-accent" style="margin-bottom:1rem">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                             color:var(--muted)">
                  📁 {uploaded.name} · {fmt_bytes(len(datos))}
                </span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                             color:var(--cyan)">
                  UTF-8 · texto
                </span>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )
    render_dashboard(stats)

    info_box(
        "🧮",
        f"<strong>H(X)</strong> = "
        f"−∑ p(xᵢ)·log₂p(xᵢ) = <strong>{stats.entropia:.6f} bits/símbolo</strong><br>"
        f"H_max para {stats.simbolos_unicos} símbolos = "
        f"<strong>{stats.entropia_maxima:.4f} bits</strong> &nbsp;·&nbsp; "
        f"Entropía relativa = <strong>{(stats.entropia/stats.entropia_maxima*100) if stats.entropia_maxima else 0:.1f}%</strong>",
    )

    # ── Symbol table ──────────────────────────────────────────
    section_label("📋", "DISTRIBUCIÓN DE SÍMBOLOS")
    render_tabla_simbolos(stats)

    # ── Compression ───────────────────────────────────────────
    section_label("⚙️", "ALGORITMO DE COMPRESIÓN")
    c_sel, c_btn = st.columns([3, 1], gap="medium")
    with c_sel:
        algo = st.selectbox(
            "Algoritmo",
            ["Huffman", "LZW", "RLE"],
            key="algo_texto",
            label_visibility="collapsed",
        )
    with c_btn:
        comprimir = st.button("▶ Comprimir", key="btn_texto")

    if comprimir:
        with st.spinner(f"Aplicando {algo}…"):
            if algo == "Huffman":
                res = CodificadorHuffman(datos).comprimir()
            elif algo == "LZW":
                res = CodificadorLZW().comprimir(datos)
            else:
                res = CodificadorRLE().comprimir(datos)

        section_label("📦", f"RESULTADO — {res.nombre_algoritmo}")
        render_resultado_compresion(res)
        render_pasos(res.pasos)

        if res.tabla_codigos:
            render_huffman_codes(res.tabla_codigos)

        if res.tabla_lzw:
            section_label("📖", "TABLA LZW (primeras entradas)")
            st.dataframe(
                pd.DataFrame(res.tabla_lzw), use_container_width=True, height=300
            )

    # ── Source code viewers ───────────────────────────────────
    with st.expander("🔎 Ver Código Fuente del Algoritmo", expanded=False):
        algo_code_map = {
            "Huffman": _CODE_HUFFMAN,
            "LZW": _CODE_LZW,
            "RLE": _CODE_RLE,
        }
        tabs_code = st.tabs(["Algoritmo seleccionado", "Entropía de Shannon"])
        with tabs_code[0]:
            st.code(algo_code_map.get(algo, ""), language="python")
        with tabs_code[1]:
            st.code(_CODE_ENTROPIA, language="python")

    render_formulas_matematicas()


# ─── TAB 2: IMAGEN ───────────────────────────────────────────────────────────

def tab_imagen() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")

    with col_up:
        section_label("🖼️", "CARGAR IMAGEN")
        uploaded = st.file_uploader(
            "Sube una imagen",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            key="uploader_imagen",
            label_visibility="collapsed",
        )

    with col_info:
        section_label("ℹ️", "ALGORITMOS DISPONIBLES")
        for algo_info in [
            ("DCT — JPEG", "Transformada discreta del coseno.\nCodificación con pérdida por bloques 8×8.", "tag-v"),
            ("RLE Imagen", "Run-Length sobre bytes de píxel.\nBase del formato BMP comprimido.", "tag-c"),
            ("Huffman Bytes", "Huffman sobre distribución de bytes.\nAnálisis entrópico de píxeles.", "tag-g"),
        ]:
            st.markdown(
                f"""<div class="algo-card">
                      <div class="ac-top">
                        <span class="ac-name">{algo_info[0]}</span>
                        <span class="ac-tag {algo_info[2]}">{'lossy' if 'DCT' in algo_info[0] else 'lossless'}</span>
                      </div>
                      <div class="ac-desc">{algo_info[1]}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    if uploaded is None:
        render_no_file(
            "🖼️",
            "Sube una imagen para analizarla",
            "Formatos: .png  .jpg  .jpeg  .bmp  .tiff",
        )
        return

    datos = uploaded.read()

    # ── Show image preview ────────────────────────────────────
    pc1, pc2 = st.columns([1, 2], gap="large")
    with pc1:
        section_label("🔍", "PREVISUALIZACIÓN")
        st.image(datos, use_container_width=True)
        if PIL_AVAILABLE:
            img = PILImage.open(io.BytesIO(datos))
            mode = img.mode
            w, h = img.size
            st.markdown(
                f"""<div class="stat-mini" style="margin-top:0.5rem">
                      <span class="sm-val">{w}×{h}</span>
                      <span class="sm-lbl">{mode} · {fmt_bytes(len(datos))}</span>
                    </div>""",
                unsafe_allow_html=True,
            )

    with pc2:
        analizador = AnalizadorImagen(datos, uploaded.name)
        with st.spinner("Calculando distribución de bytes…"):
            stats = analizador.calcular_todo()

        section_label("📊", "DASHBOARD INFORMATIVO")
        render_dashboard(stats)
        info_box(
            "💡",
            f"La entropía <strong>{stats.entropia:.4f} bits/byte</strong> indica cuánta "
            f"información contiene cada byte de la imagen.<br>"
            f"Entropía máxima posible (256 valores): <strong>8.0000 bits</strong>.<br>"
            f"Potencial de compresión sin pérdida: aprox. "
            f"<strong>{(1 - stats.entropia/8)*100:.1f}%</strong>.",
        )

    section_label("📋", "DISTRIBUCIÓN DE BYTES (PÍXELES)")
    render_tabla_simbolos(stats, max_rows=40)

    # ── Histogram ─────────────────────────────────────────────
    section_label("📈", "HISTOGRAMA DE BYTES")
    hist_data = {
        "Valor byte": list(range(256)),
        "Frecuencia": [stats.frecuencias.get(b, 0) for b in range(256)],
    }
    df_hist = pd.DataFrame(hist_data).set_index("Valor byte")
    st.bar_chart(df_hist, color="#06b6d4", height=200)

    # ── Compression ───────────────────────────────────────────
    section_label("⚙️", "ALGORITMO DE COMPRESIÓN")
    c_sel, c_q, c_btn = st.columns([2, 2, 1], gap="medium")
    with c_sel:
        algo_img = st.selectbox(
            "Algoritmo",
            ["DCT — JPEG-like", "RLE", "Huffman (bytes)"],
            key="algo_imagen",
            label_visibility="collapsed",
        )
    with c_q:
        calidad = st.slider(
            "Calidad (solo DCT)",
            min_value=1, max_value=95, value=50,
            key="calidad_dct",
            disabled=(algo_img != "DCT — JPEG-like"),
        )
    with c_btn:
        comprimir_img = st.button("▶ Comprimir", key="btn_imagen")

    if comprimir_img:
        with st.spinner(f"Aplicando {algo_img}…"):
            if algo_img == "DCT — JPEG-like":
                res = CodificadorDCT().comprimir(datos, calidad)
            elif algo_img == "RLE":
                res = CodificadorRLE().comprimir(datos)
            else:
                res = CodificadorHuffman(datos).comprimir()

        section_label("📦", f"RESULTADO — {res.nombre_algoritmo}")
        render_resultado_compresion(res)
        render_pasos(res.pasos)
        if res.tabla_codigos:
            render_huffman_codes(res.tabla_codigos)

    with st.expander("🔎 Ver Código Fuente del Algoritmo", expanded=False):
        t1, t2, t3 = st.tabs(["DCT / JPEG", "RLE", "Entropía"])
        with t1:
            st.code(_CODE_DCT, language="python")
        with t2:
            st.code(_CODE_RLE, language="python")
        with t3:
            st.code(_CODE_ENTROPIA, language="python")

    render_formulas_matematicas()


# ─── TAB 3: AUDIO ────────────────────────────────────────────────────────────

def _read_wav_pcm(datos: bytes) -> Tuple[bytes, int, int, int]:
    """Extract raw PCM from WAV bytes. Returns (pcm, n_channels, sample_width, framerate)."""
    try:
        with wave.open(io.BytesIO(datos)) as wf:
            return wf.readframes(wf.getnframes()), wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
    except Exception:
        return datos, 1, 2, 44100


def tab_audio() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")

    with col_up:
        section_label("🎵", "CARGAR ARCHIVO DE AUDIO")
        uploaded = st.file_uploader(
            "Sube un archivo de audio (.wav recomendado para análisis PCM)",
            type=["wav", "mp3", "ogg", "flac"],
            key="uploader_audio",
            label_visibility="collapsed",
        )

    with col_info:
        section_label("ℹ️", "ALGORITMOS DISPONIBLES")
        for algo_info in [
            ("μ-Law G.711", "Companding logarítmico 2:1.\nEstándar telefónico (VoIP, PSTN).", "tag-c"),
            ("ADPCM", "Codificación diferencial adaptiva 4:1.\nEstándar IMA/DVI para multimedia.", "tag-v"),
            ("Huffman PCM", "Huffman sobre distribución de muestras.\nAnálisis entrópico de audio.", "tag-g"),
        ]:
            st.markdown(
                f"""<div class="algo-card">
                      <div class="ac-top">
                        <span class="ac-name">{algo_info[0]}</span>
                        <span class="ac-tag {algo_info[2]}">lossless*</span>
                      </div>
                      <div class="ac-desc">{algo_info[1]}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    if uploaded is None:
        render_no_file(
            "🎵",
            "Sube un archivo de audio para analizarlo",
            "Recomendado: .wav PCM 16-bit · También: .mp3  .ogg  .flac",
        )
        return

    datos = uploaded.read()
    is_wav = uploaded.name.lower().endswith(".wav")

    if is_wav:
        pcm_bytes, n_ch, samp_w, framerate = _read_wav_pcm(datos)
        duracion_s = len(pcm_bytes) / (n_ch * samp_w * framerate) if framerate else 0
        info_box(
            "🔊",
            f"<strong>WAV PCM</strong> detectado · "
            f"{n_ch}ch · {samp_w*8}-bit · {framerate:,} Hz · "
            f"Duración: <strong>{duracion_s:.2f} s</strong> · "
            f"{fmt_bytes(len(pcm_bytes))} de datos PCM",
        )
        datos_analisis = pcm_bytes
    else:
        info_box(
            "⚠️",
            "Formato comprimido detectado. Se analizará el flujo de bytes en bruto.<br>"
            "Para análisis PCM completo, sube un archivo <strong>.wav PCM 16-bit</strong>.",
            "amber",
        )
        datos_analisis = datos

    analizador = AnalizadorAudio(datos_analisis, uploaded.name)
    with st.spinner("Calculando estadísticas de audio…"):
        stats = analizador.calcular_todo()

    section_label("📊", "DASHBOARD INFORMATIVO")
    render_dashboard(stats)

    info_box(
        "🧮",
        f"La entropía del audio es <strong>{stats.entropia:.4f} bits/byte</strong>.<br>"
        f"Audio PCM sin comprimir: 16 bits/muestra. Entropía efectiva: "
        f"<strong>{stats.entropia:.2f} bits</strong>.<br>"
        f"Potencial de compresión sin pérdida: "
        f"<strong>{max(0, (1 - stats.entropia/16)*100):.1f}%</strong> (aprox. para PCM 16-bit).",
    )

    section_label("📋", "DISTRIBUCIÓN DE BYTES DE AUDIO")
    render_tabla_simbolos(stats, max_rows=35)

    # ── Waveform (mini) ───────────────────────────────────────
    if is_wav and len(pcm_bytes) >= 4:
        section_label("〰️", "FORMA DE ONDA (MUESTRA)")
        n_s = len(pcm_bytes) // 2
        samples_arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        step = max(1, len(samples_arr) // 500)
        waveform = samples_arr[::step][:500]
        wf_df = pd.DataFrame({"Amplitud": waveform})
        st.line_chart(wf_df, color="#06b6d4", height=160)

    # ── Compression ───────────────────────────────────────────
    section_label("⚙️", "ALGORITMO DE COMPRESIÓN")
    c_sel, c_btn = st.columns([4, 1], gap="medium")
    with c_sel:
        algo_aud = st.selectbox(
            "Algoritmo",
            ["μ-Law G.711", "ADPCM", "Huffman (bytes)"],
            key="algo_audio",
            label_visibility="collapsed",
        )
    with c_btn:
        comprimir_aud = st.button("▶ Comprimir", key="btn_audio")

    if comprimir_aud:
        with st.spinner(f"Aplicando {algo_aud}…"):
            if algo_aud == "μ-Law G.711":
                res = CodificadorMuLaw().comprimir(datos_analisis)
            elif algo_aud == "ADPCM":
                res = CodificadorADPCM().comprimir(datos_analisis)
            else:
                res = CodificadorHuffman(datos_analisis).comprimir()

        section_label("📦", f"RESULTADO — {res.nombre_algoritmo}")
        render_resultado_compresion(res)
        render_pasos(res.pasos)
        if res.tabla_codigos:
            render_huffman_codes(res.tabla_codigos)

    with st.expander("🔎 Ver Código Fuente del Algoritmo", expanded=False):
        t1, t2, t3 = st.tabs(["μ-Law G.711", "ADPCM (IMA)", "Entropía"])
        with t1:
            st.code(_CODE_MULAW, language="python")
        with t2:
            st.code(_CODE_ADPCM, language="python")
        with t3:
            st.code(_CODE_ENTROPIA, language="python")

    render_formulas_matematicas()


# ─── TAB 4: VIDEO ────────────────────────────────────────────────────────────

def tab_video() -> None:
    col_up, col_info = st.columns([3, 2], gap="large")

    with col_up:
        section_label("🎬", "CARGAR ARCHIVO DE VIDEO")
        uploaded = st.file_uploader(
            "Sube un archivo de video",
            type=["mp4", "avi", "mkv", "mov", "webm"],
            key="uploader_video",
            label_visibility="collapsed",
        )

    with col_info:
        section_label("ℹ️", "CONCEPTOS APLICADOS")
        for concepto in [
            ("Tramas I/P/B", "Intra, Predicción, Bidireccional.\nBase de H.264/HEVC.", "tag-c"),
            ("DCT Temporal", "DCT sobre diferencias entre frames.\nReducción de redundancia temporal.", "tag-v"),
            ("Entropía Cabac", "Codificación aritmética adaptiva.\nHasta 15% mejor que Huffman.", "tag-g"),
        ]:
            st.markdown(
                f"""<div class="algo-card">
                      <div class="ac-top">
                        <span class="ac-name">{concepto[0]}</span>
                        <span class="ac-tag {concepto[2]}">concepto</span>
                      </div>
                      <div class="ac-desc">{concepto[1]}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    if uploaded is None:
        render_no_file(
            "🎬",
            "Sube un archivo de video para analizarlo",
            "Formatos: .mp4  .avi  .mkv  .mov  .webm",
        )
        return

    datos = uploaded.read()
    analizador = AnalizadorVideo(datos, uploaded.name)

    info_box(
        "ℹ️",
        f"<strong>Nota:</strong> El análisis byte-level del contenedor de video muestra la "
        f"distribución de bits del flujo comprimido.<br>"
        f"Archivo: <strong>{uploaded.name}</strong> · "
        f"Tamaño: <strong>{fmt_bytes(len(datos))}</strong><br>"
        f"Para análisis frame-a-frame usa OpenCV + esta arquitectura como backend.",
    )

    with st.spinner("Analizando flujo de bytes de video…"):
        stats = analizador.calcular_todo()

    section_label("📊", "DASHBOARD INFORMATIVO")
    render_dashboard(stats)

    info_box(
        "🔬",
        f"El flujo de video comprimido (H.264/HEVC) tiene entropía alta "
        f"(<strong>{stats.entropia:.4f} bits/byte</strong>) porque "
        f"los datos ya están codificados entrópicamente.<br>"
        f"Un video RAW tendría entropía menor y sería más compresible.",
        "violet",
    )

    section_label("📋", "DISTRIBUCIÓN DE BYTES DEL FLUJO")
    render_tabla_simbolos(stats, max_rows=40)

    # ── Byte histogram ────────────────────────────────────────
    section_label("📈", "HISTOGRAMA DE BYTES (flujo comprimido)")
    hist = {"Valor": list(range(256)), "Freq": [stats.frecuencias.get(b, 0) for b in range(256)]}
    df_h = pd.DataFrame(hist).set_index("Valor")
    st.bar_chart(df_h, color="#8b5cf6", height=180)

    # ── Video compression theory ──────────────────────────────
    section_label("🎞️", "ARQUITECTURA DE COMPRESIÓN DE VIDEO")
    info_box(
        "📐",
        """<strong>Pipeline H.264/HEVC (educativo):</strong><br>
        1. <strong>Partición en Macroblocks</strong> (16×16 píxeles por defecto)<br>
        2. <strong>Predicción Intra (tramas I)</strong>: DCT sobre bloque actual<br>
        3. <strong>Predicción Inter (tramas P/B)</strong>: Motion Estimation + DCT sobre residual<br>
        4. <strong>Cuantización</strong>: reducción de precisión de coef. DCT<br>
        5. <strong>CABAC/CAVLC</strong>: codificación entrópica de los coeficientes cuantizados<br>
        6. <strong>Deblocking Filter</strong>: suavizado de artefactos en bloques""",
    )

    # ── Metrics simulation ────────────────────────────────────
    section_label("📦", "SIMULACIÓN DE COMPRESIÓN (Teórica)")
    c1, c2, c3 = st.columns(3)
    with c1:
        factor_i = 3.5
        st.metric("Trama I (Intra)", f"{factor_i:.1f}:1", "Solo DCT espacial")
    with c2:
        factor_p = 12.0
        st.metric("Trama P (Predictiva)", f"{factor_p:.1f}:1", "DCT + Motion Vector")
    with c3:
        factor_b = 20.0
        st.metric("Trama B (Bidireccional)", f"{factor_b:.1f}:1", "Máxima compresión")

    with st.expander("🔎 Ver Código Fuente — Conceptos de Video", expanded=False):
        st.code(
            """\
# ╔══════════════════════════════════════════════════════════╗
# ║  COMPRESIÓN DE VIDEO — Conceptos H.264/HEVC              ║
# ╚══════════════════════════════════════════════════════════╝
import numpy as np

# ── Estimación de Movimiento (Motion Estimation) ──────────
def motion_estimation(frame_ref: np.ndarray, frame_curr: np.ndarray,
                       block_size: int = 16) -> list[tuple]:
    \"\"\"
    Full-search block matching.
    Para cada macroblock en frame_curr, encuentra el mejor
    matching en frame_ref dentro de un rango de búsqueda.
    
    Métrica: SAD (Sum of Absolute Differences)
    SAD(mv) = ΣΣ |frame_curr(x,y) - frame_ref(x+mvx, y+mvy)|
    \"\"\"
    H, W = frame_curr.shape
    motion_vectors = []
    for i in range(0, H - block_size, block_size):
        for j in range(0, W - block_size, block_size):
            curr_block = frame_curr[i:i+block_size, j:j+block_size]
            best_sad, best_mv = float('inf'), (0, 0)
            # Búsqueda en ventana ±16 píxeles (STUB: rango limitado)
            for dy in range(-16, 17):
                for dx in range(-16, 17):
                    ri, rj = i+dy, j+dx
                    if 0<=ri and ri+block_size<=H and 0<=rj and rj+block_size<=W:
                        ref_block = frame_ref[ri:ri+block_size, rj:rj+block_size]
                        sad = np.sum(np.abs(curr_block.astype(int) - ref_block))
                        if sad < best_sad:
                            best_sad, best_mv = sad, (dx, dy)
            motion_vectors.append(best_mv)
    return motion_vectors   # lista de (mvx, mvy) por macroblock

# ── Codificación del Residual ──────────────────────────────
def encode_residual(original: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    \"\"\"
    Residual = Original − Predicho
    El residual tiene menor energía → más compresible con DCT.
    \"\"\"
    return original.astype(int) - predicted.astype(int)

# ── Tipos de tramas ────────────────────────────────────────
# Trama I (Intra):   codificada sin referencia a otros frames
#                    Solo DCT espacial sobre cada macroblock
# Trama P (Pred):    referencia el frame I más reciente
#                    Motion vector + DCT del residual
# Trama B (Bi-dir):  referencia frames I y P anteriores y futuros
#                    Mayor compresión pero más complejidad
# 
# Estructura típica GOP (Group Of Pictures):
#   I B B P B B P B B I ...  (GOP size = 12 en H.264)
""",
            language="python",
        )

    render_formulas_matematicas()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    inject_css()
    render_header()

    tabs = st.tabs(["📄  Texto", "🖼️  Imagen", "🎵  Audio", "🎬  Video"])

    with tabs[0]:
        tab_texto()
    with tabs[1]:
        tab_imagen()
    with tabs[2]:
        tab_audio()
    with tabs[3]:
        tab_video()

    # ── Footer ─────────────────────────────────────────────────
    st.markdown(
        """
        <div style="margin-top:3rem;padding:1rem 0;border-top:1px solid var(--border);
                    text-align:center;font-family:'IBM Plex Mono',monospace;
                    font-size:0.65rem;color:var(--muted)">
          DataLab · Compresión de Datos · Laboratorio de Teoría de la Información &nbsp;·&nbsp;
          Shannon (1948) · Huffman (1952) · LZW (1977–1984) · JPEG (1992) · G.711 (1972)
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
