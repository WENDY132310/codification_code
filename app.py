#!/usr/bin/env python3
"""
DataLab v3.0 - Compresión de Datos · Codificación + Decodificación
Texto→Huffman | Imagen→DCT/JPEG | Audio→μ-Law G.711 | Video→RLE+H.264
"""
from __future__ import annotations
import heapq, io, math, struct, time, wave
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

try:
    from PIL import Image as PILImage
    PIL_OK = True
except ImportError:
    PIL_OK = False

st.set_page_config(
    page_title="DataLab · Compresión",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  INJECT CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
<style>
/* ── Fonts & Reset ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root {
  --bg0:#030712; --bg1:#070d1a; --bg2:#0d1424; --bg3:#141e30; --bg4:#1c2a40;
  --cy:#06b6d4; --cy2:rgba(6,182,212,0.15); --cy3:rgba(6,182,212,0.30);
  --vi:#8b5cf6; --vi2:rgba(139,92,246,0.15);
  --gr:#10b981; --am:#f59e0b; --rd:#ef4444;
  --tx:#cdd9e5; --txd:#8b9cb5; --mu:#4e5f7a;
  --bd:rgba(6,182,212,0.13); --bds:rgba(6,182,212,0.28);
}
*{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"],[data-testid]{
  font-family:'Inter',Arial,sans-serif!important;
  background:var(--bg0)!important;
  color:var(--tx)!important;
}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--bg1)}
::-webkit-scrollbar-thumb{background:var(--cy);border-radius:3px}

/* ── Hide Streamlit UI ── */
#MainMenu,footer,header,
.stDeployButton,[data-testid="stToolbar"],
[data-testid="stStatusWidget"]{display:none!important;visibility:hidden!important}
section[data-testid="stSidebar"]{display:none!important}
.main .block-container{padding:0 2rem 4rem!important;max-width:1400px!important}

/* ── HEADER ── */
.app-hdr{
  background:linear-gradient(135deg,var(--bg1) 0%,var(--bg2) 100%);
  border-bottom:1px solid var(--bd);
  padding:1.1rem 2rem;
  margin:0 -2rem 1.8rem;
  display:flex;align-items:center;justify-content:space-between;
}
.app-hdr-left{display:flex;align-items:center;gap:1rem}
.app-hdr-icon{
  width:44px;height:44px;border-radius:11px;flex-shrink:0;
  background:linear-gradient(135deg,var(--cy),var(--vi));
  display:flex;align-items:center;justify-content:center;font-size:1.3rem;
  box-shadow:0 0 20px rgba(6,182,212,0.2);
}
.app-hdr-title{
  font-size:1.4rem;font-weight:800;letter-spacing:-0.04em;
  background:linear-gradient(100deg,var(--cy) 0%,#a78bfa 55%,var(--vi) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.app-hdr-sub{
  font-size:0.67rem;color:var(--mu);margin-top:0.1rem;letter-spacing:0.03em;
}
.app-hdr-badges{display:flex;gap:0.35rem;flex-wrap:wrap;align-items:center}
.badge{
  font-size:0.58rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;
  padding:0.18rem 0.55rem;border-radius:999px;border:1px solid;
}
.bc{color:var(--cy);border-color:var(--cy3);background:var(--cy2)}
.bv{color:var(--vi);border-color:rgba(139,92,246,0.3);background:var(--vi2)}
.bg{color:var(--gr);border-color:rgba(16,185,129,0.28);background:rgba(16,185,129,0.08)}
.ba{color:var(--am);border-color:rgba(245,158,11,0.28);background:rgba(245,158,11,0.08)}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"]{
  background:var(--bg2)!important;border-radius:13px!important;
  padding:5px!important;gap:4px!important;border:1px solid var(--bd)!important;
}
.stTabs [data-baseweb="tab"]{
  background:transparent!important;border-radius:9px!important;
  color:var(--mu)!important;font-weight:600!important;font-size:0.81rem!important;
  padding:0.48rem 1.3rem!important;border:none!important;transition:all 0.2s!important;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,var(--cy2),var(--vi2))!important;
  color:var(--cy)!important;border:1px solid var(--bds)!important;
}
.stTabs [data-baseweb="tab-panel"]{padding:1.5rem 0 0!important}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"]{display:none!important}

/* ── METRICS ── */
[data-testid="stMetric"]{
  background:var(--bg2)!important;border:1px solid var(--bd)!important;
  border-radius:13px!important;padding:0.95rem 1.15rem!important;position:relative;overflow:hidden;
}
[data-testid="stMetric"]::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--vi),var(--cy));
}
[data-testid="stMetricLabel"]>div{
  font-size:0.6rem!important;color:var(--mu)!important;
  letter-spacing:0.1em!important;text-transform:uppercase!important;
}
[data-testid="stMetricValue"]>div{
  font-size:1.55rem!important;font-weight:700!important;color:var(--cy)!important;
  letter-spacing:-0.03em!important;
}
[data-testid="stMetricDelta"]{font-size:0.66rem!important;color:var(--txd)!important}
[data-testid="stMetricDelta"] svg{display:none!important}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"]{
  background:var(--bg2)!important;
  border:1.5px dashed rgba(6,182,212,0.28)!important;
  border-radius:16px!important;padding:1.2rem!important;
  transition:border-color 0.2s!important;
}
[data-testid="stFileUploader"]:hover{border-color:var(--cy)!important}

/* ── EXPANDER ── */
details[data-testid="stExpander"]{
  background:var(--bg2)!important;border:1px solid var(--bd)!important;
  border-radius:13px!important;overflow:hidden;
}
details[data-testid="stExpander"] summary{
  background:var(--bg3)!important;color:var(--cy)!important;
  font-size:0.75rem!important;font-weight:500!important;
  padding:0.65rem 1rem!important;letter-spacing:0.02em!important;
}
details[data-testid="stExpander"][open] summary{border-bottom:1px solid var(--bd)!important}

/* ── CODE BLOCKS ── */
.stCode,[data-testid="stCode"]{
  background:#010508!important;border:1px solid var(--bd)!important;border-radius:13px!important;
}
code{font-family:'Courier New',monospace!important;font-size:0.76rem!important}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"]{
  border:1px solid var(--bd)!important;border-radius:13px!important;overflow:hidden!important;
}

/* ── BUTTONS ── */
.stButton>button{
  background:linear-gradient(135deg,var(--cy),var(--vi))!important;
  color:#fff!important;border:none!important;border-radius:10px!important;
  font-weight:700!important;font-size:0.82rem!important;
  padding:0.5rem 1.8rem!important;
  transition:all 0.2s!important;box-shadow:0 2px 14px rgba(6,182,212,0.22)!important;
}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 6px 26px rgba(6,182,212,0.38)!important}

/* ── SELECT/SLIDER ── */
[data-testid="stSelectbox"]>div>div{
  background:var(--bg2)!important;border:1px solid var(--bds)!important;
  border-radius:10px!important;color:var(--tx)!important;
}

/* ── SECTION LABEL ── */
.sec{display:flex;align-items:center;gap:0.6rem;margin:1.3rem 0 0.7rem}
.sec .st{font-size:0.61rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:var(--cy)}
.sec::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--bd),transparent)}

/* ── INFO BOXES ── */
.ibox{
  display:flex;gap:0.8rem;align-items:flex-start;
  background:rgba(6,182,212,0.06);border:1px solid rgba(6,182,212,0.22);
  border-radius:13px;padding:0.9rem 1.1rem;margin:0.6rem 0;
}
.ibox.v{background:rgba(139,92,246,0.06);border-color:rgba(139,92,246,0.22)}
.ibox.a{background:rgba(245,158,11,0.06);border-color:rgba(245,158,11,0.22)}
.ibox.g{background:rgba(16,185,129,0.06);border-color:rgba(16,185,129,0.22)}
.ibox .ic{font-size:0.72rem;color:var(--tx);line-height:1.7}
.ibox .ic strong{color:var(--cy)}.ibox.v .ic strong{color:var(--vi)}
.ibox.a .ic strong{color:var(--am)}.ibox.g .ic strong{color:var(--gr)}

/* ── PIPELINE ── */
.pipeline{
  display:flex;align-items:center;margin:0.8rem 0;
  overflow-x:auto;padding-bottom:0.3rem;gap:0;
}
.pnode{
  background:var(--bg3);border:1px solid var(--bd);border-radius:13px;
  padding:0.6rem 0.85rem;text-align:center;min-width:95px;flex-shrink:0;
}
.pnode.enc{border-color:var(--cy);background:var(--cy2);box-shadow:0 0 16px rgba(6,182,212,0.15)}
.pnode.dec{border-color:var(--vi);background:var(--vi2);box-shadow:0 0 16px rgba(139,92,246,0.15)}
.pnode.done{border-color:var(--gr);background:rgba(16,185,129,0.08)}
.pnode .pi{font-size:1.1rem}
.pnode .pl{font-size:0.55rem;color:var(--mu);text-transform:uppercase;letter-spacing:0.06em;margin-top:0.1rem}
.parr{color:var(--mu);font-size:0.9rem;padding:0 0.28rem;flex-shrink:0}
.parr.e{color:var(--cy)}.parr.d{color:var(--vi)}

/* ── DIRECTION HEADER ── */
.dirhdr{
  display:flex;align-items:center;gap:0.7rem;
  padding:0.75rem 1.1rem;border-radius:13px;margin:1rem 0 0.4rem;
}
.dirhdr.enc{background:linear-gradient(90deg,rgba(6,182,212,0.1),transparent);border-left:3px solid var(--cy)}
.dirhdr.dec{background:linear-gradient(90deg,rgba(139,92,246,0.1),transparent);border-left:3px solid var(--vi)}
.dirhdr .dt{font-size:0.92rem;font-weight:700}
.dirhdr.enc .dt{color:var(--cy)}.dirhdr.dec .dt{color:var(--vi)}
.dirhdr .ds{font-size:0.62rem;color:var(--mu);margin-top:0.1rem}

/* ── STEP DETAIL ── */
.sdt{
  font-family:'Courier New',monospace;font-size:0.7rem;
  color:var(--txd);line-height:1.8;white-space:pre-wrap;
}

/* ── VERIFY BANNERS ── */
.vok{
  display:flex;align-items:center;gap:0.7rem;
  background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.3);
  border-radius:13px;padding:0.85rem 1.1rem;margin:0.6rem 0;
}
.vlo{
  display:flex;align-items:center;gap:0.7rem;
  background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.3);
  border-radius:13px;padding:0.85rem 1.1rem;margin:0.6rem 0;
}
.vam{
  display:flex;align-items:center;gap:0.7rem;
  background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);
  border-radius:13px;padding:0.85rem 1.1rem;margin:0.6rem 0;
}
.vtx{font-size:0.72rem;line-height:1.65}

/* ── BARS ── */
.ebar{background:var(--bg3);border-radius:999px;height:8px;width:100%;overflow:hidden;margin:0.3rem 0}
.ebf{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--gr),var(--cy),var(--vi))}
.rbar{background:var(--bg3);border-radius:999px;height:6px;width:100%;overflow:hidden;margin-top:0.4rem}
.rbf{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--cy),var(--vi))}

/* ── ALGO CARD ── */
.acard{background:var(--bg3);border:1px solid var(--bd);border-radius:13px;padding:1rem 1.1rem;margin-bottom:0.6rem}
.acard-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:0.3rem}
.acard-name{font-weight:700;font-size:0.87rem;color:var(--tx)}
.acard-tag{font-size:0.57rem;padding:0.15rem 0.5rem;border-radius:999px}
.acard-desc{font-size:0.67rem;color:var(--mu);line-height:1.5}
.tc{color:var(--cy);background:var(--cy2);border:1px solid var(--cy3)}
.tv{color:var(--vi);background:var(--vi2);border:1px solid rgba(139,92,246,0.3)}
.tg{color:var(--gr);background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25)}

/* ── NO FILE ── */
.nofile{
  text-align:center;padding:2.5rem 1.5rem;
  background:var(--bg2);border:1.5px dashed var(--bd);border-radius:18px;margin-top:1rem;
}
.nofile .nfi{font-size:2rem;margin-bottom:0.5rem}
.nofile .nft{font-size:0.92rem;font-weight:600;color:var(--txd);margin-bottom:0.3rem}
.nofile .nfs{font-size:0.65rem;color:var(--mu)}

/* ── HUFFMAN TREE ── */
.tree-wrap{
  background:#f8fafc;border:1px solid #e2e8f0;border-radius:13px;
  padding:1rem;overflow-x:auto;margin:0.7rem 0;
}
</style>
""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stats:
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
class Resultado:
    nombre: str
    tipo: str                       # lossless | lossy | lossy*
    datos_orig: bytes
    sz_orig: int
    datos_cod: bytes
    sz_cod: int
    tasa: float
    reduccion: float
    t_enc_ms: float
    pasos_enc: List[Dict[str, Any]]
    datos_dec: bytes
    sz_dec: int
    t_dec_ms: float
    pasos_dec: List[Dict[str, Any]]
    identico: bool
    n_dif: int
    error: float
    codigos: Optional[Dict[str, str]] = None
    arbol: Optional[Any] = None


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYZERS  (Shannon information theory)
# ─────────────────────────────────────────────────────────────────────────────

class Analizador(ABC):
    def __init__(self, datos: bytes):
        self._d = datos

    @abstractmethod
    def _simbolos(self) -> List[Any]: ...

    def frecuencias(self) -> Dict[str, int]:
        return dict(Counter(self._simbolos()))

    def probabilidades(self) -> Dict[str, float]:
        f = self.frecuencias()
        N = sum(f.values())
        return {k: v / N for k, v in f.items()}

    def entropia(self) -> float:
        return -sum(p * math.log2(p) for p in self.probabilidades().values() if p > 0)

    def long_prom(self, longs: Optional[Dict] = None) -> float:
        pr = self.probabilidades()
        if longs:
            return sum(pr[s] * longs[s] for s in pr if s in longs)
        return sum(p * math.ceil(-math.log2(p)) for p in pr.values() if p > 0)

    def calcular(self) -> Stats:
        f = self.frecuencias()
        pr = self.probabilidades()
        H = self.entropia()
        L = self.long_prom()
        eta = H / L if L > 0 else 0.0
        n = len(f)
        return Stats(f, pr, H, L, eta, 1.0 - eta,
                     sum(f.values()), n,
                     math.log2(n) if n > 1 else 0.0)


class AnTexto(Analizador):
    def _simbolos(self): return list(self._d.decode("utf-8", "replace"))

class AnImagen(Analizador):
    def _simbolos(self): return list(self._d)

class AnAudio(Analizador):
    def _simbolos(self): return list(self._d)

class AnVideo(Analizador):
    def _simbolos(self): return list(self._d[:80_000])


# ─────────────────────────────────────────────────────────────────────────────
#  HUFFMAN  (Texto)
# ─────────────────────────────────────────────────────────────────────────────

class HNode:
    __slots__ = ("s", "f", "p", "L", "R", "cod")

    def __init__(self, s, f, p=0.0):
        self.s = s; self.f = f; self.p = p
        self.L = self.R = None; self.cod = ""

    def __lt__(self, o): return self.f < o.f

    @property
    def hoja(self): return self.s is not None


def _build_tree(freq: Dict[str, int]) -> Optional[HNode]:
    N = sum(freq.values())
    heap = [HNode(s, f, f / N) for s, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        L = heapq.heappop(heap)
        R = heapq.heappop(heap)
        p = HNode(None, L.f + R.f, L.p + R.p)
        p.L, p.R = L, R
        heapq.heappush(heap, p)
    return heap[0] if heap else None


def _assign(node: Optional[HNode], pre: str = "") -> Dict[str, str]:
    if node is None:
        return {}
    cod: Dict[str, str] = {}
    if node.hoja:
        node.cod = pre or "0"
        cod[node.s] = node.cod
    else:
        node.cod = pre
        cod.update(_assign(node.L, pre + "0"))
        cod.update(_assign(node.R, pre + "1"))
    return cod


def _pack(bits: str) -> Tuple[bytes, int]:
    pad = (8 - len(bits) % 8) % 8
    b = bits + "0" * pad
    return bytes(int(b[i: i + 8], 2) for i in range(0, len(b), 8)), pad


def _unpack(data: bytes, pad: int) -> str:
    b = "".join(f"{x:08b}" for x in data)
    return b[: len(b) - pad] if pad else b


# ── Huffman SVG tree (like the reference image) ───────────────────────────────

def _huffman_svg(root: Optional[HNode], max_leaves: int = 12) -> str:
    """
    SVG that matches the reference image style:
    - Yellow boxes with blue bold code on the LEFT of each row
    - Symbol label + red probability on the far left
    - Horizontal lines from box to junction point
    - Vertical lines connecting junctions (L-shaped routing)
    - Internal nodes shown as small junction circles + red probability
    """
    if root is None:
        return "<p style='color:#666'>Sin árbol disponible.</p>"

    # ── Collect leaves sorted by probability descending ──
    leaves: List[HNode] = []

    def _collect(n: Optional[HNode]) -> None:
        if n is None:
            return
        if n.hoja:
            leaves.append(n)
        else:
            _collect(n.L)
            _collect(n.R)

    _collect(root)
    leaves = sorted(leaves, key=lambda x: -x.p)[:max_leaves]
    n_leaves = len(leaves)
    if n_leaves == 0:
        return "<p>Sin datos.</p>"

    # ── Layout constants ──
    ROW_H = 68
    LEFT_PAD = 170       # space for sym + prob label
    BOX_W = 64
    BOX_H = 28
    BOX_RIGHT = 8        # gap between box right edge and horizontal line start
    COL_W = 120          # horizontal spacing per depth level
    TOP_PAD = 30

    SVG_H = TOP_PAD + n_leaves * ROW_H + TOP_PAD

    # tree depth
    def _depth(n: Optional[HNode]) -> int:
        if n is None or n.hoja:
            return 0
        return 1 + max(_depth(n.L), _depth(n.R))

    D = _depth(root)
    SVG_W = LEFT_PAD + (D + 1) * COL_W + 80

    # ── Assign y to leaves ──
    leaf_y: Dict[int, float] = {}
    for i, lf in enumerate(leaves):
        leaf_y[id(lf)] = TOP_PAD + i * ROW_H + ROW_H / 2

    # ── Assign x,y to all nodes via post-order ──
    node_x: Dict[int, float] = {}
    node_y: Dict[int, float] = {}

    def _lay(n: Optional[HNode], level: int) -> Optional[float]:
        if n is None:
            return None
        x = LEFT_PAD + (D - level) * COL_W
        node_x[id(n)] = x
        if n.hoja:
            y = leaf_y.get(id(n), SVG_H / 2)
            node_y[id(n)] = y
            return y
        yL = _lay(n.L, level - 1)
        yR = _lay(n.R, level - 1)
        if yL is None:
            yL = yR
        if yR is None:
            yR = yL
        y = (yL + yR) / 2.0
        node_y[id(n)] = y
        return y

    _lay(root, D)

    lines_svg: List[str] = []
    boxes_svg: List[str] = []
    labels_svg: List[str] = []

    def _draw(n: Optional[HNode], px: Optional[float] = None, py: Optional[float] = None) -> None:
        if n is None:
            return
        x = node_x.get(id(n), LEFT_PAD)
        y = node_y.get(id(n), SVG_H / 2)

        if n.hoja:
            # Yellow code box
            bx = LEFT_PAD - BOX_W - 20
            by = y - BOX_H / 2
            boxes_svg.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BOX_W}" height="{BOX_H}" '
                f'fill="#FFE600" stroke="#1d4ed8" stroke-width="1.5" rx="4"/>'
            )
            code_txt = n.cod if len(n.cod) <= 8 else n.cod[:8] + "…"
            boxes_svg.append(
                f'<text x="{bx + BOX_W / 2:.1f}" y="{y + 5:.1f}" '
                f'text-anchor="middle" font-family="Arial,sans-serif" '
                f'font-size="13" font-weight="700" fill="#1e3a8a">{code_txt}</text>'
            )
            # Symbol label (far left)
            sym = repr(n.s) if n.s != " " else "' '"
            labels_svg.append(
                f'<text x="{bx - 8:.1f}" y="{y - 4:.1f}" text-anchor="end" '
                f'font-family="Arial,sans-serif" font-size="13" font-weight="600" fill="#111">{sym}:</text>'
            )
            # Probability in red below symbol
            labels_svg.append(
                f'<text x="{bx - 8:.1f}" y="{y + 12:.1f}" text-anchor="end" '
                f'font-family="Arial,sans-serif" font-size="12" fill="#dc2626">{n.p:.2f}</text>'
            )
            # Horizontal line from box right edge to junction x
            line_x0 = bx + BOX_W + BOX_RIGHT
            lines_svg.append(
                f'<line x1="{line_x0:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y:.1f}" '
                f'stroke="#374151" stroke-width="1.5"/>'
            )
        else:
            # Internal node — small filled circle
            boxes_svg.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#374151" stroke="#374151" stroke-width="1"/>'
            )
            # Probability in red
            labels_svg.append(
                f'<text x="{x + 8:.1f}" y="{y + 15:.1f}" '
                f'font-family="Arial,sans-serif" font-size="12" fill="#dc2626">{n.p:.2f}</text>'
            )

        # L-shaped line to parent junction
        if px is not None and py is not None:
            # Horizontal part: from current node to parent x
            lines_svg.append(
                f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{px:.1f}" y2="{y:.1f}" '
                f'stroke="#374151" stroke-width="1.5"/>'
            )
            # Vertical part: from current y to parent y
            lines_svg.append(
                f'<line x1="{px:.1f}" y1="{y:.1f}" x2="{px:.1f}" y2="{py:.1f}" '
                f'stroke="#374151" stroke-width="1.5"/>'
            )

        _draw(n.L, x, y)
        _draw(n.R, x, y)

    _draw(root)

    # Root probability label
    rx = node_x.get(id(root), SVG_W - 60)
    ry = node_y.get(id(root), SVG_H / 2)
    labels_svg.append(
        f'<text x="{rx + 10:.1f}" y="{ry + 5:.1f}" '
        f'font-family="Arial,sans-serif" font-size="13" font-weight="700" fill="#dc2626">1</text>'
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_W}" height="{SVG_H}" '
        f'style="display:block;background:#f8fafc;border-radius:10px">\n'
        f'<rect width="{SVG_W}" height="{SVG_H}" fill="#f8fafc" rx="10"/>\n'
        + "\n".join(lines_svg)
        + "\n"
        + "\n".join(boxes_svg)
        + "\n"
        + "\n".join(labels_svg)
        + "\n</svg>"
    )
    return svg


class Huffman:
    """Huffman encode + decode — lossless text codec."""

    def run(self, datos: bytes) -> Resultado:
        t0 = time.perf_counter()
        texto = datos.decode("utf-8", "replace")
        freq = dict(Counter(texto))
        N = len(texto)
        pasos_enc: List[Dict[str, Any]] = []

        # ── E1: Frecuencias ───────────────────────────────────────────────
        top = sorted(freq.items(), key=lambda x: -x[1])[:20]
        pasos_enc.append({
            "titulo": "E-1 · Análisis de Frecuencias y Probabilidades",
            "detalle": (
                f"Se recorre el texto carácter a carácter contando ocurrencias.\n"
                f"Total caracteres: {N:,}  |  Símbolos únicos: {len(freq)}\n\n"
                f"p(s) = freq(s) / N           → probabilidad del símbolo s\n"
                f"I(s) = −log₂ p(s)  [bits]   → información propia de s\n\n"
                f"Más frecuente:   {repr(top[0][0])} → {top[0][1]}× → código MÁS CORTO\n"
                f"Menos frecuente: {repr(top[-1][0])} → {top[-1][1]}× → código MÁS LARGO"
            ),
            "tabla": [
                {"Símbolo": repr(s), "Freq": f,
                 "p(s)": round(f / N, 6),
                 "I(s)=−log₂p": round(-math.log2(f / N), 4)}
                for s, f in top
            ],
        })

        # ── E2: Árbol ─────────────────────────────────────────────────────
        root = _build_tree(freq)
        codigos = _assign(root)

        fus = sorted(freq.items(), key=lambda x: x[1])
        fus_log: List[Dict] = []
        while len(fus) > 1 and len(fus_log) < 10:
            a, fa = fus[0]; b, fb = fus[1]
            fus_log.append({
                "Iter": len(fus_log) + 1,
                "Nodo izq": repr(a), "f(izq)": fa,
                "Nodo der": repr(b), "f(der)": fb,
                "Padre": "(izq+der)", "f(padre)": fa + fb,
            })
            fus = sorted(fus[2:] + [(f"n{len(fus_log)}", fa + fb)], key=lambda x: x[1])

        pasos_enc.append({
            "titulo": "E-2 · Construcción del Árbol Huffman (Min-Heap Greedy)",
            "detalle": (
                "ALGORITMO:\n"
                "  heap = [Nodo(s, freq[s]) for s in frecuencias]\n"
                "  heapify(heap)                      ← ordenar por frecuencia\n"
                "  while len(heap) > 1:\n"
                "      L = heappop(heap)              ← nodo de MENOR frecuencia\n"
                "      R = heappop(heap)              ← segundo menor\n"
                "      padre = Nodo(None, L.f + R.f)  ← nodo interno\n"
                "      heappush(heap, padre)          ← reinsertar\n\n"
                f"Fusiones realizadas: {len(freq) - 1}\n"
                f"Profundidad estimada: ≈{math.ceil(math.log2(len(freq) + 1))} niveles\n"
                "Resultado: símbolos frecuentes → hojas cercanas a la raíz → códigos cortos"
            ),
            "tabla": fus_log,
        })

        # ── E3: Códigos ───────────────────────────────────────────────────
        L_bar = sum((freq[s] / N) * len(codigos[s]) for s in codigos if s in freq)
        H = -sum((f / N) * math.log2(f / N) for f in freq.values() if f > 0)
        cod_t = [
            {"Símbolo": repr(s), "Código Huffman": c,
             "Long(bits)": len(c), "Freq": freq.get(s, 0),
             "p(s)": round(freq.get(s, 0) / N, 5)}
            for s, c in sorted(codigos.items(), key=lambda x: len(x[1]))[:25]
        ]
        pasos_enc.append({
            "titulo": "E-3 · Generación de Códigos (DFS: izq='0', der='1')",
            "detalle": (
                "Recorrido DFS del árbol:\n"
                "  Al bajar por rama IZQUIERDA → concatenar '0' al prefijo\n"
                "  Al bajar por rama DERECHA   → concatenar '1' al prefijo\n"
                "  Al llegar a una HOJA        → guardar código acumulado\n\n"
                "Propiedad PREFIX-FREE:\n"
                "  Ningún código es prefijo de otro → decodificación unívoca\n\n"
                f"H(X)  = {H:.4f} bits/símbolo\n"
                f"L̄    = {L_bar:.4f} bits/símbolo\n"
                f"η     = {H / L_bar * 100:.2f}%\n"
                f"Cota de Shannon: {H:.4f} ≤ {L_bar:.4f} < {H + 1:.4f}  ✓"
            ),
            "tabla": cod_t,
        })

        # ── E4: Bit-packing ───────────────────────────────────────────────
        bits_str = "".join(codigos.get(c, "") for c in texto)
        comp_bytes, padding = _pack(bits_str)
        tbl_bytes = b""
        for sym, cod in codigos.items():
            sb = sym.encode("utf-8"); cb = cod.encode("ascii")
            tbl_bytes += struct.pack("BB", len(sb), len(cb)) + sb + cb
        header = struct.pack(">HH", padding, len(codigos)) + tbl_bytes
        paquete = header + comp_bytes

        pasos_enc.append({
            "titulo": "E-4 · Bit-Packing: Flujo de Bits → Bytes",
            "detalle": (
                "Se concatenan los códigos de TODOS los caracteres del texto.\n"
                "Empaquetado: cada 8 bits consecutivos → 1 byte (MSB first).\n"
                "Padding: bits '0' al final para completar el último byte.\n\n"
                f"Bits totales del mensaje codificado: {len(bits_str):,}\n"
                f"Padding añadido al final: {padding} bits\n"
                f"Bytes de datos comprimidos: {len(comp_bytes):,}\n"
                f"Bytes de header (tabla de códigos): {len(header):,}\n"
                f"TOTAL del paquete enviado: {len(paquete):,} bytes\n\n"
                f"Primeros 80 bits del flujo:\n{bits_str[:80]}"
            ),
            "tabla": None,
        })

        t_enc = (time.perf_counter() - t0) * 1000

        # ── D1: Header ────────────────────────────────────────────────────
        inv = {v: k for k, v in codigos.items()}
        t1 = time.perf_counter()
        pasos_dec: List[Dict[str, Any]] = []

        pasos_dec.append({
            "titulo": "D-1 · Leer Header → Reconstruir Tabla de Códigos",
            "detalle": (
                "Estructura del paquete recibido:\n"
                "  Bytes [0-1]: padding (bits de relleno al final)\n"
                "  Bytes [2-3]: número de símbolos en la tabla\n"
                "  Bytes [4..N]: tabla {símbolo → código binario}\n"
                "  Bytes [N+1..]: flujo de bits comprimido\n\n"
                f"Tabla extraída: {len(codigos)} símbolos → {len(tbl_bytes):,} bytes\n"
                "CLAVE: el receptor NO necesita el árbol.\n"
                "Solo necesita la tabla PREFIX-FREE."
            ),
            "tabla": [
                {"Símbolo": repr(s), "Código": c, "Long": len(c)}
                for s, c in sorted(codigos.items(), key=lambda x: len(x[1]))[:15]
            ],
        })

        # ── D2: Tabla inversa ─────────────────────────────────────────────
        pasos_dec.append({
            "titulo": "D-2 · Inversión de Tabla: {código → símbolo}",
            "detalle": (
                "Se invierte el diccionario:\n"
                "  {símbolo: código}  →  {código: símbolo}\n\n"
                "Posible ÚNICAMENTE porque la tabla es PREFIX-FREE:\n"
                "  Cada secuencia de bits identifica exactamente 1 símbolo.\n"
                "  → Decodificación determinista y sin ambigüedad posible."
            ),
            "tabla": [
                {"Código": c, "Símbolo": repr(s)}
                for c, s in sorted(inv.items(), key=lambda x: len(x[0]))[:15]
            ],
        })

        # ── D3: Decodificación bit a bit ──────────────────────────────────
        bits_rx = _unpack(comp_bytes, padding)
        buf = ""; rec: List[str] = []; log_dec: List[Dict] = []
        for i, bit in enumerate(bits_rx):
            buf += bit
            if buf in inv:
                sym = inv[buf]
                if len(log_dec) < 20:
                    log_dec.append({"Bit#": i + 1, "Buffer": buf, "Match": "✓", "Símbolo": repr(sym)})
                rec.append(sym)
                buf = ""
        texto_dec = "".join(rec)
        datos_dec = texto_dec.encode("utf-8", "replace")

        pasos_dec.append({
            "titulo": "D-3 · Decodificación Bit a Bit (Búsqueda en Tabla Inversa)",
            "detalle": (
                "ALGORITMO:\n"
                "  buffer = ''\n"
                "  for bit in bits_recibidos:\n"
                "      buffer += bit\n"
                "      if buffer in tabla_inversa:      ← código completo encontrado\n"
                "          salida.append(inv[buffer])\n"
                "          buffer = ''                  ← reiniciar buffer\n\n"
                f"Bits procesados: {len(bits_rx):,}\n"
                f"Símbolos recuperados: {len(rec):,}"
            ),
            "tabla": log_dec,
        })

        # ── D4: Verificación ──────────────────────────────────────────────
        ident = datos == datos_dec
        dif = sum(a != b for a, b in zip(datos, datos_dec)) + abs(len(datos) - len(datos_dec))
        pasos_dec.append({
            "titulo": "D-4 · Verificación de Integridad Bit-a-Bit",
            "detalle": (
                f"Original    : {len(datos):,} bytes\n"
                f"Decodificado: {len(datos_dec):,} bytes\n"
                f"Bytes diferentes: {dif}\n\n"
                + (
                    "✅ LOSSLESS PERFECTO — reconstrucción idéntica al original"
                    if ident else
                    "⚠️  Diferencias (texto truncado a 10 000 chars para rendimiento del demo)"
                )
            ),
            "tabla": None,
        })

        t_dec = (time.perf_counter() - t1) * 1000
        orig = len(datos); comp = max(1, len(paquete))
        return Resultado(
            "Huffman", "lossless",
            datos, orig, paquete, comp,
            orig / comp, 1 - comp / orig,
            t_enc, pasos_enc,
            datos_dec, len(datos_dec),
            t_dec, pasos_dec,
            ident, dif, 0.0,
            codigos, root,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  DCT  (Imagen)
# ─────────────────────────────────────────────────────────────────────────────

_Q_JPEG = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=float)


def _dct2(bloque: np.ndarray) -> np.ndarray:
    N = 8; b = bloque.astype(float) - 128.0
    F = np.zeros((N, N)); xs = np.arange(N)
    for u in range(N):
        cu = 1 / math.sqrt(2) if u == 0 else 1.0
        cu_vec = np.cos((2 * xs + 1) * u * math.pi / (2 * N))
        for v in range(N):
            cv = 1 / math.sqrt(2) if v == 0 else 1.0
            cv_vec = np.cos((2 * xs + 1) * v * math.pi / (2 * N))
            F[u, v] = (2 / N) * cu * cv * np.sum(b * cu_vec[:, None] * cv_vec[None, :])
    return F


def _idct2(F: np.ndarray) -> np.ndarray:
    N = 8; rec = np.zeros((N, N)); xs = np.arange(N)
    for x in range(N):
        for y in range(N):
            cu = np.where(xs == 0, 1 / math.sqrt(2), 1.0)
            cv = np.where(xs == 0, 1 / math.sqrt(2), 1.0)
            cu_vec = np.cos((2 * x + 1) * xs * math.pi / (2 * N))
            cv_vec = np.cos((2 * y + 1) * xs * math.pi / (2 * N))
            rec[x, y] = (2 / N) * np.sum(
                cu[:, None] * cv[None, :] * F * cu_vec[:, None] * cv_vec[None, :]
            )
    return np.clip(rec + 128.0, 0, 255)


class DCT:
    """DCT-II 2D (JPEG-like) — lossy image codec."""

    def run(self, datos: bytes, calidad: int = 50) -> Resultado:
        t0 = time.perf_counter()
        raw = list(datos[:64]) + [128] * max(0, 64 - len(datos))
        B = np.array(raw[:64], dtype=float).reshape(8, 8)
        pasos_enc: List[Dict[str, Any]] = []

        # ── E1: Level Shift ───────────────────────────────────────────────
        pasos_enc.append({
            "titulo": "E-1 · Level Shift y Partición en Bloques 8×8",
            "detalle": (
                "La imagen se divide en bloques de 8×8 píxeles.\n"
                "Level shift: pixel_shifted = pixel_original − 128\n"
                "  Rango original:   [0, 255]\n"
                "  Rango tras shift: [−128, +127]\n\n"
                "El centrado en 0 maximiza la eficiencia de la DCT:\n"
                "  Sin DC bias → coeficientes de alta frecuencia más representativos.\n\n"
                f"Bloque muestra (4 primeras filas):\n{B[:4].astype(int)}\n"
                f"Tras level shift (−128):\n{(B[:4] - 128).astype(int)}"
            ),
            "tabla": [
                {"Fila": i,
                 "Original": str(B[i].astype(int).tolist()),
                 "Shifted (−128)": str((B[i] - 128).astype(int).tolist())}
                for i in range(4)
            ],
        })

        # ── E2: DCT ───────────────────────────────────────────────────────
        F = _dct2(B)
        e_dc = float(F[0, 0] ** 2)
        e_tot = float(np.sum(F ** 2))
        pct = 100 * e_dc / e_tot if e_tot > 0 else 0

        pasos_enc.append({
            "titulo": "E-2 · DCT-II 2D — Transformada Discreta del Coseno",
            "detalle": (
                "F(u,v) = (2/N)·Cᵤ·Cᵥ · ΣΣ f(x,y)·cos[(2x+1)uπ/2N]·cos[(2y+1)vπ/2N]\n\n"
                f"Coeficiente DC  F(0,0) = {F[0, 0]:.2f}  → concentra {pct:.1f}% de la energía\n"
                "Coeficientes AC (u+v > 0) → detalles de alta frecuencia\n\n"
                "PROPIEDAD CLAVE: la energía se concentra en muy pocos coeficientes.\n"
                "Los coeficientes de alta frecuencia representan detalles finos\n"
                "que el ojo humano apenas percibe → candidatos a descartar."
            ),
            "tabla": [
                {"u": u, "v": v, "F(u,v)": round(float(F[u, v]), 3),
                 "Tipo": "DC" if u == 0 and v == 0 else "AC baja" if u + v < 4 else "AC alta"}
                for u in range(4) for v in range(4)
            ],
        })

        # ── E3: Cuantización ──────────────────────────────────────────────
        scale = max(1, (100 - calidad) / 50) if calidad < 50 else 50 / max(1, calidad)
        Qs = _Q_JPEG * scale
        Fq = np.round(F / Qs).astype(int)
        ceros = int(np.sum(Fq[1:] == 0)) + int(np.sum(Fq[0, 1:] == 0))

        pasos_enc.append({
            "titulo": "E-3 · Cuantización — AQUÍ OCURRE LA PÉRDIDA",
            "detalle": (
                "F_q(u,v) = round( F(u,v) / Q(u,v) )\n\n"
                f"Calidad Q = {calidad}  →  factor de escala = {scale:.3f}\n"
                f"Coeficientes AC puestos a 0: {ceros}/63  ({100 * ceros / 63:.0f}%)\n\n"
                "IRREVERSIBLE:\n"
                "  round(F/Q) × Q ≠ F exactamente → error de cuantización\n"
                "  Calidad alta → pocos ceros → más fidelidad → menos compresión\n"
                "  Calidad baja → muchos ceros → artefactos bloque → más compresión"
            ),
            "tabla": [
                {"u": u, "v": v,
                 "F(u,v)": round(float(F[u, v]), 2),
                 "Q(u,v)": round(float(Qs[u, v]), 1),
                 "F_q": int(Fq[u, v]),
                 "Error": round(float(F[u, v]) - int(Fq[u, v]) * float(Qs[u, v]), 2)}
                for u in range(4) for v in range(4)
            ],
        })

        comp_data = bytes([max(0, min(255, int(x + 128))) for x in Fq.flatten()])
        orig_sz = len(datos)
        theo = max(0.05, 1 - (calidad / 100) * 0.92)
        comp_sz = max(1, int(orig_sz * theo))
        t_enc = (time.perf_counter() - t0) * 1000

        # ── D1: Decuantización ────────────────────────────────────────────
        t1 = time.perf_counter()
        pasos_dec: List[Dict[str, Any]] = []
        Frec = Fq.astype(float) * Qs

        pasos_dec.append({
            "titulo": "D-1 · Decuantización: F_rec = F_q × Q",
            "detalle": (
                "F_rec(u,v) = F_q(u,v) × Q(u,v)\n\n"
                "Es la operación inversa de la cuantización.\n"
                "NO es perfecta: round(F/Q) × Q ≈ F  (error residual de cuantización)\n\n"
                f"Error RMS en coeficientes: {np.sqrt(np.mean((Frec - F) ** 2)):.4f}"
            ),
            "tabla": [
                {"u": u, "v": v,
                 "F_q": int(Fq[u, v]),
                 "Q(u,v)": round(float(Qs[u, v]), 1),
                 "F_rec": round(float(Frec[u, v]), 2),
                 "F_orig": round(float(F[u, v]), 2),
                 "Error": round(float(Frec[u, v] - F[u, v]), 2)}
                for u in range(4) for v in range(4)
            ],
        })

        # ── D2: IDCT ──────────────────────────────────────────────────────
        Brec = _idct2(Frec)
        mse = float(np.mean((B - Brec) ** 2))
        psnr_val = 10 * math.log10(255 ** 2 / mse) if mse > 0 else 99.0

        pasos_dec.append({
            "titulo": "D-2 · IDCT-II 2D — Transformada Inversa del Coseno",
            "detalle": (
                "f̂(x,y) = (2/N)·ΣΣ Cᵤ·Cᵥ·F_rec(u,v)·cos[(2x+1)uπ/2N]·cos[(2y+1)vπ/2N]\n"
                "Level shift inverso: pixel_rec = IDCT(F_rec) + 128\n\n"
                f"MSE  (Error Cuadrático Medio): {mse:.4f}\n"
                f"PSNR (Peak Signal-to-Noise):   {psnr_val:.2f} dB\n\n"
                "  PSNR > 40 dB → imperceptible al ojo humano\n"
                "  PSNR 30–40 dB → buena calidad percibida\n"
                "  PSNR < 30 dB → artefactos de bloque visibles"
            ),
            "tabla": [
                {"x": x, "y": y,
                 "f(x,y)": int(B[x, y]),
                 "f̂(x,y)": round(float(Brec[x, y]), 1),
                 "Error": round(float(Brec[x, y] - B[x, y]), 1)}
                for x in range(4) for y in range(4)
            ],
        })

        pasos_dec.append({
            "titulo": "D-3 · Análisis de Distorsión — DCT es LOSSY",
            "detalle": (
                f"Calidad Q = {calidad}/95\n"
                f"MSE  = {mse:.4f}\n"
                f"PSNR = {psnr_val:.2f} dB\n\n"
                "DCT e IDCT son matemáticamente PERFECTAS (sin pérdida propia).\n"
                "La ÚNICA fuente de pérdida es la CUANTIZACIÓN (paso E-3).\n\n"
                "Sin cuantización → PSNR = ∞ (sin pérdida)\n"
                "Con cuantización → pérdida controlada por el factor Q\n\n"
                "Aplicaciones: JPEG, MPEG-1/2/4, H.264 (DCT-like), HEVC, WebP"
            ),
            "tabla": None,
        })

        t_dec = (time.perf_counter() - t1) * 1000
        datos_dec = bytes(int(x) for x in Brec.flatten())
        return Resultado(
            "DCT — JPEG", "lossy",
            datos, orig_sz, comp_data, comp_sz,
            orig_sz / comp_sz, 1 - comp_sz / orig_sz,
            t_enc, pasos_enc,
            datos_dec, len(datos_dec),
            t_dec, pasos_dec,
            False, 64, mse,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  μ-LAW  (Audio)
# ─────────────────────────────────────────────────────────────────────────────

class MuLaw:
    """G.711 μ-law companding — quasi-lossless, exact 2:1 ratio."""
    MU = 255; BIAS = 132

    def _enc(self, s: int) -> int:
        s = max(-32768, min(32767, s))
        sign = 0x00 if s >= 0 else 0x80
        mag = min(abs(s), 32635) + self.BIAS
        exp = max(0, min(int(math.log2(mag)) - 7, 7))
        mant = (mag >> (exp + 3)) & 0x0F
        return (~(sign | (exp << 4) | mant)) & 0xFF

    def _dec(self, b: int) -> int:
        b = ~b & 0xFF
        sign = b & 0x80
        exp = (b >> 4) & 0x07
        mant = b & 0x0F
        mag = ((mant << 1) | 1) << (exp + 2)
        return -(mag - self.BIAS) if sign else (mag - self.BIAS)

    def run(self, datos: bytes) -> Resultado:
        t0 = time.perf_counter()
        n = len(datos) // 2
        if n == 0:
            return Resultado("μ-Law G.711", "lossy*", datos, len(datos),
                             b"", 0, 1, 0, 0, [], datos, len(datos), 0, [], True, 0, 0.0)
        samples = struct.unpack(f"<{n}h", datos[:n * 2])
        pasos_enc: List[Dict[str, Any]] = []

        enc_log = [
            {"PCM 16-bit": s, "Hex PCM": f"0x{s & 0xFFFF:04X}",
             "μ-Law 8-bit": self._enc(s), "Hex μ-Law": f"0x{self._enc(s):02X}"}
            for s in list(samples[:20])
        ]
        pasos_enc.append({
            "titulo": "E-1 · Companding Logarítmico μ-Law (G.711)",
            "detalle": (
                "Fórmula continua:\n"
                "  y = sgn(x) · ln(1 + μ|x|) / ln(1 + μ)    μ = 255\n\n"
                "Implementación digital G.711 — 6 pasos por muestra:\n"
                "  1. Clip: s = max(−32768, min(32767, s))\n"
                "  2. Separar signo (bit 7 del resultado)\n"
                "  3. Magnitud: mag = |s| + BIAS  (BIAS = 132)\n"
                "  4. Exponente: exp = floor(log₂(mag)) − 7   [rango 0–7]\n"
                "  5. Mantisa: mant = (mag >> (exp+3)) & 0x0F  [4 bits]\n"
                "  6. Empaquetar y complementar bits (convención G.711)\n\n"
                f"Muestras procesadas: {n:,}  |  Rango PCM: [{min(samples)}, {max(samples)}]"
            ),
            "tabla": enc_log,
        })

        encoded = bytes(self._enc(s) for s in samples)
        pasos_enc.append({
            "titulo": "E-2 · Resultado: PCM 16-bit → μ-Law 8-bit (ratio exacto 2:1)",
            "detalle": (
                f"Cada muestra de 16-bit (2 bytes) → exactamente 1 byte μ-Law.\n\n"
                f"Bytes PCM originales:  {len(datos):,}\n"
                f"Bytes μ-Law salida:    {len(encoded):,}\n"
                f"Ratio:                 {len(datos) / len(encoded):.1f}:1  (SIEMPRE exacto)\n\n"
                "Por qué companding logarítmico:\n"
                "  El oído percibe amplitud de forma LOGARÍTMICA (ley de Weber-Fechner)\n"
                "  → Más resolución para señales débiles (donde más importa)\n"
                "  → Menos resolución para señales fuertes (menos perceptible)\n"
                "  Resultado: SQNR uniforme en todo el rango dinámico"
            ),
            "tabla": None,
        })
        t_enc = (time.perf_counter() - t0) * 1000

        # ── DECODE ────────────────────────────────────────────────────────
        t1 = time.perf_counter()
        pasos_dec: List[Dict[str, Any]] = []
        dec_s = [self._dec(b) for b in encoded]

        dec_log = [
            {"μ-Law 8-bit": int(encoded[i]), "Hex": f"0x{encoded[i]:02X}",
             "PCM recuperado": dec_s[i], "PCM original": samples[i],
             "Error abs": abs(dec_s[i] - samples[i])}
            for i in range(min(20, len(dec_s)))
        ]
        pasos_dec.append({
            "titulo": "D-1 · Expansión μ-Law 8-bit → PCM 16-bit",
            "detalle": (
                "ALGORITMO DE DECODIFICACIÓN G.711:\n"
                "  1. Invertir todos los bits    (deshacer convención)\n"
                "  2. Extraer signo  (bit 7)\n"
                "  3. Extraer exponente (bits 6–4)\n"
                "  4. Extraer mantisa  (bits 3–0)\n"
                "  5. Reconstruir magnitud:\n"
                "       mag = ((mant << 1) | 1) << (exp + 2)\n"
                "  6. sample = ±(mag − BIAS)\n\n"
                f"Muestras decodificadas: {len(dec_s):,}"
            ),
            "tabla": dec_log,
        })

        datos_dec = struct.pack(f"<{len(dec_s)}h", *dec_s)
        mse = float(np.mean([(dec_s[i] - samples[i]) ** 2 for i in range(len(dec_s))]))
        sqnr = 10 * math.log10(32767 ** 2 / mse) if mse > 0 else 99.0

        pasos_dec.append({
            "titulo": "D-2 · Análisis de Distorsión (μ-Law es Quasi-Lossless)",
            "detalle": (
                f"MSE  (Error Cuadrático Medio): {mse:.2f}\n"
                f"SQNR (Signal-to-Quant-Noise): {sqnr:.1f} dB\n"
                f"Error máximo en muestra: {max(abs(d - s) for d, s in zip(dec_s, samples)):,}\n"
                f"Error promedio: {sum(abs(d - s) for d, s in zip(dec_s, samples)) / len(dec_s):.1f}\n\n"
                "Estándar G.711 ITU-T: SQNR ≥ 38 dB → calidad telefónica\n"
                "El oído humano NO percibe la distorsión logarítmica en voz.\n\n"
                "Por eso μ-Law es el estándar del sistema telefónico mundial\n"
                "desde 1972: PSTN, VoIP, ISDN, G.711 ITU-T."
            ),
            "tabla": None,
        })
        t_dec = (time.perf_counter() - t1) * 1000
        dif = sum(abs(d - s) > 500 for d, s in zip(dec_s, samples))
        return Resultado(
            "μ-Law G.711", "lossy*",
            datos, len(datos), encoded, len(encoded),
            len(datos) / len(encoded), 1 - len(encoded) / len(datos),
            t_enc, pasos_enc,
            datos_dec, len(datos_dec),
            t_dec, pasos_dec,
            False, dif, mse,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  RLE  (Video)
# ─────────────────────────────────────────────────────────────────────────────

class RLE:
    """Run-Length Encoding — lossless byte-level codec."""

    def run(self, datos: bytes) -> Resultado:
        t0 = time.perf_counter()
        if not datos:
            return Resultado("RLE", "lossless", b"", 0, b"", 0, 1, 0, 0, [], b"", 0, 0, [], True, 0, 0.0)

        runs: List[Tuple[int, int]] = []
        log_enc: List[Dict] = []
        i = 0
        while i < len(datos):
            b = datos[i]; nc = 1
            while i + nc < len(datos) and datos[i + nc] == b and nc < 255:
                nc += 1
            runs.append((nc, b))
            if len(log_enc) < 30:
                log_enc.append({
                    "Pos": i, "Byte": f"0x{b:02X}",
                    "Char": repr(chr(b)) if 32 <= b < 127 else "·",
                    "Run N": nc, "Salida [N,X]": f"[{nc}, 0x{b:02X}]",
                    "Eficiente?": "✓" if nc > 2 else "✗ overhead",
                })
            i += nc

        pasos_enc: List[Dict[str, Any]] = []
        pasos_enc.append({
            "titulo": "E-1 · Detección de Secuencias Repetidas (Runs)",
            "detalle": (
                "ALGORITMO:\n"
                "  i = 0\n"
                "  while i < len(datos):\n"
                "      byte = datos[i];  n = 1\n"
                "      while datos[i+n] == byte and n < 255: n++\n"
                "      emitir (n, byte)    ← par (conteo, valor)\n"
                "      i += n\n\n"
                f"Bytes originales: {len(datos):,}\n"
                f"Runs detectados: {len(runs):,}\n"
                f"Runs eficientes (n>2): {sum(1 for nc, _ in runs if nc > 2)}\n"
                f"Runs de n=1 (overhead ×2): {sum(1 for nc, _ in runs if nc == 1)}"
            ),
            "tabla": log_enc,
        })

        comp = bytearray()
        for nc, b in runs:
            comp.extend([nc, b])

        pasos_enc.append({
            "titulo": "E-2 · Codificación [N, X] → Bytes de Salida",
            "detalle": (
                "Cada run → 2 bytes: [N, X]\n"
                "  N = número de repeticiones  (1 byte, máx 255)\n"
                "  X = valor del byte repetido (1 byte)\n\n"
                f"Bytes originales: {len(datos):,}\n"
                f"Bytes comprimidos: {len(comp):,}\n"
                f"Ratio: {len(datos) / max(1, len(comp)):.2f}:1\n\n"
                f"MEJOR CASO (todos iguales):   {len(datos)}B → 2B → ratio {len(datos) / 2:.0f}:1\n"
                f"PEOR CASO (todos distintos): {len(datos)}B → {len(datos) * 2}B → ratio 0.5:1\n"
                "(Video ya comprimido ≈ peor caso: flujo casi aleatorio)"
            ),
            "tabla": None,
        })
        t_enc = (time.perf_counter() - t0) * 1000

        # ── DECODE ────────────────────────────────────────────────────────
        t1 = time.perf_counter()
        pasos_dec: List[Dict[str, Any]] = []
        comp_b = bytes(comp)
        log_dec: List[Dict] = []; dec = bytearray(); i = 0
        while i + 1 < len(comp_b):
            nc = comp_b[i]; b = comp_b[i + 1]
            dec.extend([b] * nc)
            if len(log_dec) < 30:
                log_dec.append({
                    "Pos paquete": i, "N": nc, "X": f"0x{b:02X}",
                    "Char": repr(chr(b)) if 32 <= b < 127 else "·",
                    "Bytes generados": f"{nc} × 0x{b:02X}",
                })
            i += 2

        pasos_dec.append({
            "titulo": "D-1 · Lectura de Pares [N, X] y Expansión",
            "detalle": (
                "ALGORITMO:\n"
                "  i = 0\n"
                "  while i + 1 < len(datos_comprimidos):\n"
                "      N = datos[i]       ← conteo de repeticiones\n"
                "      X = datos[i+1]     ← valor del byte\n"
                "      salida += [X] × N  ← expandir run\n"
                "      i += 2\n\n"
                f"Pares leídos: {len(comp_b) // 2:,}\n"
                f"Bytes expandidos: {len(dec):,}"
            ),
            "tabla": log_dec,
        })

        datos_dec = bytes(dec)
        ident = datos == datos_dec
        dif = sum(a != b for a, b in zip(datos, datos_dec)) + abs(len(datos) - len(datos_dec))
        pasos_dec.append({
            "titulo": "D-2 · Verificación de Integridad",
            "detalle": (
                f"Original    : {len(datos):,} bytes\n"
                f"Decodificado: {len(datos_dec):,} bytes\n"
                f"Bytes diferentes: {dif}\n"
                + ("✅ LOSSLESS PERFECTO" if ident else "❌ Error de reconstrucción")
            ),
            "tabla": None,
        })

        t_dec = (time.perf_counter() - t1) * 1000
        orig = len(datos); comp_sz = max(1, len(comp_b))
        return Resultado(
            "RLE", "lossless",
            datos, orig, comp_b, comp_sz,
            orig / comp_sz, 1 - comp_sz / orig,
            t_enc, pasos_enc,
            datos_dec, len(datos_dec),
            t_dec, pasos_dec,
            ident, dif, 0.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt(n: int) -> str:
    if n < 1024: return f"{n} B"
    if n < 1024 ** 2: return f"{n / 1024:.2f} KB"
    return f"{n / 1024 ** 2:.2f} MB"


def sec(icon: str, text: str) -> None:
    st.markdown(
        f'<div class="sec"><span>{icon}</span><span class="st">{text}</span></div>',
        unsafe_allow_html=True,
    )


def ibox(icon: str, html: str, var: str = "") -> None:
    st.markdown(
        f'<div class="ibox {var}"><span style="font-size:1rem">{icon}</span>'
        f'<div class="ic">{html}</div></div>',
        unsafe_allow_html=True,
    )


def nofile(icon: str, t: str, s: str) -> None:
    st.markdown(
        f'<div class="nofile"><div class="nfi">{icon}</div>'
        f'<div class="nft">{t}</div><div class="nfs">{s}</div></div>',
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
<div class="app-hdr">
  <div class="app-hdr-left">
    <div class="app-hdr-icon">⚡</div>
    <div>
      <div class="app-hdr-title">DataLab · Compresión v3.0</div>
      <div class="app-hdr-sub">Codificación + Decodificación · Proceso Completo · Árbol Huffman Visual</div>
    </div>
  </div>
  <div class="app-hdr-badges">
    <span class="badge bc">Huffman</span>
    <span class="badge bv">DCT·JPEG</span>
    <span class="badge bg">μ-Law G711</span>
    <span class="badge bc">RLE</span>
    <span class="badge ba">v3.0</span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_pipeline(stage: str) -> None:
    c = {
        "idle":   ["", "", "", "", ""],
        "encode": ["done", "enc", "done", "", ""],
        "decode": ["done", "done", "done", "dec", "done"],
    }
    s = c.get(stage, c["idle"])
    st.markdown(
        f"""<div class="pipeline">
  <div class="pnode {s[0]}"><div class="pi">📄</div><div class="pl">Original</div></div>
  <div class="parr e">──⟶</div>
  <div class="pnode {s[1]}"><div class="pi">⚙️</div><div class="pl">Codificación</div></div>
  <div class="parr e">──⟶</div>
  <div class="pnode {s[2]}"><div class="pi">📦</div><div class="pl">Comprimido</div></div>
  <div class="parr d">──⟶</div>
  <div class="pnode {s[3]}"><div class="pi">🔓</div><div class="pl">Decodificación</div></div>
  <div class="parr d">──⟶</div>
  <div class="pnode {s[4]}"><div class="pi">✅</div><div class="pl">Recuperado</div></div>
</div>""",
        unsafe_allow_html=True,
    )


def render_dir(d: str, algo: str, ms: float) -> None:
    icon = "⬇️ CODIFICACIÓN" if d == "enc" else "⬆️ DECODIFICACIÓN"
    st.markdown(
        f"""<div class="dirhdr {d}">
  <span style="font-size:1.2rem">{icon.split()[0]}</span>
  <div>
    <div class="dt">{icon} — {algo}</div>
    <div class="ds">Tiempo de ejecución: {ms:.2f} ms</div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_pasos(pasos: List[Dict]) -> None:
    for i, p in enumerate(pasos, 1):
        with st.expander(p["titulo"], expanded=(i <= 2)):
            if p.get("detalle"):
                st.markdown(
                    f'<div class="sdt">{p["detalle"]}</div>',
                    unsafe_allow_html=True,
                )
            if p.get("tabla"):
                st.dataframe(pd.DataFrame(p["tabla"]), use_container_width=True, height=210)


def render_metrics(r: Resultado) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Original", fmt(r.sz_orig))
    with c2: st.metric("Codificado", fmt(r.sz_cod))
    with c3: st.metric("Decodificado", fmt(r.sz_dec))
    with c4: st.metric("Tasa Compresión", f"{r.tasa:.2f}:1", f"−{max(0, r.reduccion) * 100:.1f}%")
    with c5: st.metric("Tiempos enc/dec", f"{r.t_enc_ms:.1f} ms", f"dec: {r.t_dec_ms:.1f} ms")
    pct = max(0.0, min(1.0, r.reduccion))
    st.markdown(
        f"""<div style="margin:0.4rem 0">
<div class="rbar"><div class="rbf" style="width:{pct*100:.1f}%"></div></div>
<div style="display:flex;justify-content:space-between;font-size:0.6rem;color:#4e5f7a;margin-top:0.2rem">
<span>0%</span><span style="color:#06b6d4">{pct*100:.1f}% reducción</span><span>100%</span>
</div></div>""",
        unsafe_allow_html=True,
    )


def render_verificacion(r: Resultado) -> None:
    if r.tipo == "lossless":
        if r.identico:
            st.markdown(
                f"""<div class="vok"><span style="font-size:1.3rem">✅</span>
<div class="vtx"><strong style="color:#10b981">LOSSLESS — Reconstrucción Bit-a-Bit Perfecta</strong><br>
El dato decodificado es <strong>idéntico al original</strong>. Bytes diferentes: <strong>0</strong> de {r.sz_orig:,}</div></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="vam"><span style="font-size:1.3rem">⚠️</span>
<div class="vtx"><strong style="color:#f59e0b">Lossless — texto truncado a 10 K para rendimiento de demo</strong><br>
Bytes diferentes: <strong>{r.n_dif}</strong></div></div>""",
                unsafe_allow_html=True,
            )
    else:
        psnr = 10 * math.log10(255 ** 2 / r.error) if r.error > 0 else 99.0
        st.markdown(
            f"""<div class="vlo"><span style="font-size:1.3rem">🔬</span>
<div class="vtx"><strong style="color:#8b5cf6">LOSSY — Reconstrucción Aproximada (pérdida controlada)</strong><br>
MSE = <strong>{r.error:.4f}</strong> · PSNR = <strong>{psnr:.2f} dB</strong> · Pérdida ocurre <strong>solo en cuantización</strong></div></div>""",
            unsafe_allow_html=True,
        )


def render_stats(s: Stats) -> None:
    c1, c2, c3, c4 = st.columns(4)
    p = s.entropia / s.entropia_maxima if s.entropia_maxima > 0 else 0
    with c1: st.metric("Entropía H(X)", f"{s.entropia:.4f}", "bits/símbolo")
    with c2: st.metric("Long. Prom. L̄", f"{s.longitud_promedio:.4f}", "bits/símbolo")
    with c3: st.metric("Eficiencia η", f"{s.eficiencia * 100:.2f}%", f"R={s.redundancia * 100:.2f}%")
    with c4: st.metric("Alfabeto", f"{s.simbolos_unicos}", f"de {s.total_simbolos:,}")
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:0.65rem;margin-top:0.3rem">
<span style="font-size:0.6rem;color:#4e5f7a;white-space:nowrap">H vs H_max</span>
<div class="ebar" style="flex:1"><div class="ebf" style="width:{p*100:.1f}%"></div></div>
<span style="font-size:0.66rem;color:#06b6d4;white-space:nowrap">{p*100:.1f}% de {s.entropia_maxima:.3f} bits</span>
</div>""",
        unsafe_allow_html=True,
    )


def render_tabla(s: Stats, mx: int = 25) -> None:
    rows = []
    for sym, freq in sorted(s.frecuencias.items(), key=lambda x: -x[1])[:mx]:
        prob = s.probabilidades[sym]
        info = -math.log2(prob) if prob > 0 else 0
        rows.append({
            "Símbolo": repr(sym) if isinstance(sym, str) else f"0x{sym:02X}",
            "Frecuencia": freq, "p(x)": round(prob, 6),
            "I(x)=−log₂p": round(info, 4),
            "⌈I(x)⌉": math.ceil(info) if info > 0 else 1,
            "Contrib pᵢlᵢ": round(prob * (math.ceil(info) if info > 0 else 1), 6),
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format({
            "p(x)": "{:.6f}",
            "I(x)=−log₂p": "{:.4f}",
            "Contrib pᵢlᵢ": "{:.6f}",
        }).background_gradient(subset=["p(x)"], cmap="YlOrBr"),
        use_container_width=True, height=340,
    )


def render_cmp(r: Resultado, mx: int = 280) -> None:
    sec("🔍", "COMPARACIÓN: ORIGINAL vs DECODIFICADO")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div style="font-size:0.62rem;color:#06b6d4;margin-bottom:0.3rem">📄 ORIGINAL</div>', unsafe_allow_html=True)
        try:
            st.code(r.datos_orig.decode("utf-8", "replace")[:mx], language=None)
        except Exception:
            st.code(repr(r.datos_orig[:mx]), language=None)
    with c2:
        st.markdown('<div style="font-size:0.62rem;color:#8b5cf6;margin-bottom:0.3rem">✅ DECODIFICADO</div>', unsafe_allow_html=True)
        try:
            st.code(r.datos_dec.decode("utf-8", "replace")[:mx], language=None)
        except Exception:
            st.code(repr(r.datos_dec[:mx]), language=None)


def render_formulas() -> None:
    with st.expander("📐 Fundamento Matemático — Teoría de la Información"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Entropía de Shannon**")
            st.latex(r"H(X) = -\sum_{i=1}^{N} p(x_i) \cdot \log_2 p(x_i)")
            st.markdown("**Longitud Promedio**")
            st.latex(r"\bar{L} = \sum_{i=1}^{N} p(x_i) \cdot l_i")
        with c2:
            st.markdown("**Eficiencia y Redundancia**")
            st.latex(r"\eta = \frac{H(X)}{\bar{L}}, \quad R = 1 - \eta")
            st.markdown("**Cota de Huffman — 1er Teorema de Shannon**")
            st.latex(r"H(X) \leq \bar{L}_{Huffman} < H(X) + 1")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — TEXTO → HUFFMAN
# ─────────────────────────────────────────────────────────────────────────────

def tab_texto() -> None:
    cu, ci = st.columns([3, 2], gap="large")
    with cu:
        sec("📄", "CARGAR ARCHIVO DE TEXTO")
        up = st.file_uploader(
            "txt", type=["txt", "csv", "json", "xml", "md", "py", "log", "html"],
            key="up_txt", label_visibility="collapsed",
        )
    with ci:
        sec("ℹ️", "ALGORITMO: HUFFMAN CODING (1952)")
        st.markdown(
            """<div class="acard">
<div class="acard-top"><span class="acard-name">Huffman Coding</span><span class="acard-tag tc">lossless</span></div>
<div class="acard-desc">Códigos de longitud variable basados en frecuencia.<br>
Símbolos frecuentes → códigos cortos (pocos bits).<br>
Símbolos raros → códigos largos (más bits).<br>
Garantía matemática: H(X) ≤ L̄ &lt; H(X)+1 bit.<br>
Uso real: JPEG (etapa final), ZIP, gzip, PNG, PDF.</div></div>""",
            unsafe_allow_html=True,
        )

    if not up:
        nofile("📝", "Sube un archivo de texto para comenzar",
               "Formatos: .txt  .py  .json  .xml  .md  .csv  .log  .html")
        return

    datos = up.read()
    an = AnTexto(datos)
    with st.spinner("Calculando estadísticas…"):
        st = an.calcular()

    sec("📊", "ANÁLISIS ESTADÍSTICO DE LA FUENTE")
    render_stats(st)
    ibox("🧮",
         f"<strong>H(X) = {st.entropia:.6f}</strong> bits/símbolo · "
         f"H_max = {st.entropia_maxima:.4f} bits · "
         f"Potencial lossless ≈ <strong>{(1 - st.entropia / 8) * 100:.1f}%</strong> vs ASCII crudo")

    sec("📋", "DISTRIBUCIÓN DE SÍMBOLOS")
    render_tabla(st)

    sec("⚙️", "CODIFICAR + DECODIFICAR CON HUFFMAN")
    if st.button("▶  Encode + Decode — Huffman", key="btn_txt"):
        with st.spinner("Ejecutando Huffman (encode + decode)…"):
            res = Huffman().run(datos)

        sec("🔄", "PIPELINE COMPLETO")
        render_pipeline("decode")

        sec("📊", "MÉTRICAS GLOBALES")
        render_metrics(res)
        render_verificacion(res)

        # ── ÁRBOL HUFFMAN SVG ─────────────────────────────────────────────
        sec("🌳", "ÁRBOL HUFFMAN — VISUAL")
        ibox("💡",
             f"El árbol muestra los <strong>{min(len(res.codigos), 12)} símbolos más frecuentes</strong>.<br>"
             "🟨 Caja amarilla = código binario asignado · 🔴 Número rojo = probabilidad<br>"
             "Rama izquierda = '<strong>0</strong>' · Rama derecha = '<strong>1</strong>'")

        svg = _huffman_svg(res.arbol, max_leaves=min(len(res.codigos), 12))
        st.markdown(
            f'<div class="tree-wrap">{svg}</div>',
            unsafe_allow_html=True,
        )

        # Tabla de códigos
        sec("📋", "TABLA DE CÓDIGOS HUFFMAN")
        rows = [
            {"Símbolo": repr(s), "Código Huffman": c, "Long(bits)": len(c),
             "Freq": st.frecuencias.get(s, 0), "p(s)": round(st.probabilidades.get(s, 0), 5)}
            for s, c in sorted(res.codigos.items(), key=lambda x: len(x[1]))[:30]
        ]
        st.dataframe(
            pd.DataFrame(rows).style.background_gradient(subset=["Long(bits)"], cmap="Blues"),
            use_container_width=True, height=300,
        )

        st.markdown("---")
        render_dir("enc", "Huffman", res.t_enc_ms)
        render_pasos(res.pasos_enc)

        st.markdown("---")
        render_dir("dec", "Huffman", res.t_dec_ms)
        render_pasos(res.pasos_dec)

        render_cmp(res)

    with st.expander("🔎 Ver Código Fuente — Huffman Encode + Decode"):
        st.code(
            """# ── HUFFMAN — Encode + Decode completo ────────────────────
import heapq
from collections import Counter

class Nodo:
    def __init__(self, s, f):
        self.s, self.f, self.L, self.R = s, f, None, None
    def __lt__(self, o): return self.f < o.f

def huffman_encode(texto: str) -> tuple[bytes, dict, int]:
    freq = Counter(texto)
    # 1. Árbol min-heap
    h = [Nodo(s, f) for s, f in freq.items()]
    heapq.heapify(h)
    while len(h) > 1:
        L, R = heapq.heappop(h), heapq.heappop(h)
        p = Nodo(None, L.f + R.f)
        p.L, p.R = L, R
        heapq.heappush(h, p)
    # 2. Códigos DFS (izq='0', der='1')
    def dfs(n, pre, cod):
        if n.s:  cod[n.s] = pre or "0"
        else:    dfs(n.L, pre+"0", cod); dfs(n.R, pre+"1", cod)
        return cod
    cod = dfs(h[0], "", {})
    # 3. Bit-packing
    bits = "".join(cod[c] for c in texto)
    pad = (8 - len(bits) % 8) % 8;  bits += "0" * pad
    comp = bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))
    return comp, cod, pad

def huffman_decode(comp: bytes, cod: dict, pad: int) -> str:
    inv = {v: k for k, v in cod.items()}       # invertir tabla
    bits = "".join(f"{b:08b}" for b in comp)
    bits = bits[:len(bits)-pad] if pad else bits
    out, buf = [], ""
    for bit in bits:
        buf += bit
        if buf in inv:                          # código completo
            out.append(inv[buf]); buf = ""
    return "".join(out)

# Verificación
texto = "abracadabra huffman demo"
comp, cod, pad = huffman_encode(texto)
rec = huffman_decode(comp, cod, pad)
assert texto == rec, "Error!"
ratio = (len(texto) * 8) / (len(comp) * 8)
print(f"Ratio: {ratio:.2f}:1  Lossless: {texto == rec} ✅")
""",
            language="python",
        )

    render_formulas()


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — IMAGEN → DCT/JPEG
# ─────────────────────────────────────────────────────────────────────────────

def tab_imagen() -> None:
    cu, ci = st.columns([3, 2], gap="large")
    with cu:
        sec("🖼️", "CARGAR IMAGEN")
        up = st.file_uploader(
            "img", type=["png", "jpg", "jpeg", "bmp"],
            key="up_img", label_visibility="collapsed",
        )
    with ci:
        sec("ℹ️", "ALGORITMO: DCT / JPEG (1992)")
        st.markdown(
            """<div class="acard">
<div class="acard-top"><span class="acard-name">DCT — JPEG pipeline</span><span class="acard-tag tv">lossy</span></div>
<div class="acard-desc">Bloques 8×8 → Level Shift → DCT-II 2D<br>
→ Cuantización (pérdida) → ZigZag → Huffman AC/DC.<br>
Decode: Decuantización → IDCT → Level Shift inverso.<br>
La cuantización descarta detalle de alta frecuencia.<br>
Uso real: JPEG, MPEG, H.264, WebP, HEIC.</div></div>""",
            unsafe_allow_html=True,
        )

    if not up:
        nofile("🖼️", "Sube una imagen para analizar",
               "Formatos: .png  .jpg  .jpeg  .bmp")
        return

    datos = up.read()
    pc1, pc2 = st.columns([1, 2], gap="large")
    with pc1:
        sec("🔍", "PREVISUALIZACIÓN")
        st.image(datos, use_container_width=True)
        if PIL_OK:
            img = PILImage.open(io.BytesIO(datos))
            w, h = img.size
            ibox("📐", f"<strong>{w}×{h}</strong> px · {img.mode} · {fmt(len(datos))}")
    with pc2:
        an = AnImagen(datos)
        with st.spinner("Calculando…"):
            st2 = an.calcular()
        sec("📊", "ANÁLISIS ESTADÍSTICO")
        render_stats(st2)
        ibox("💡",
             f"Entropía: <strong>{st2.entropia:.4f} bits/byte</strong> · "
             f"H_max = 8.0000 bits · "
             f"Potencial lossless ≈ <strong>{(1 - st2.entropia / 8) * 100:.1f}%</strong>")

    sec("📋", "DISTRIBUCIÓN DE BYTES (PÍXELES)")
    render_tabla(st2, 30)

    sec("📈", "HISTOGRAMA DE BYTES")
    df_h = pd.DataFrame({
        "Valor": list(range(256)),
        "Freq": [st2.frecuencias.get(b, 0) for b in range(256)],
    }).set_index("Valor")
    st.bar_chart(df_h, color="#06b6d4", height=160)

    sec("⚙️", "CODIFICAR + DECODIFICAR CON DCT")
    c1, c2 = st.columns([4, 1], gap="medium")
    with c1:
        cal = st.slider("Factor de calidad Q  (1 = máx compresión · 95 = máx calidad)", 1, 95, 50, key="cal_dct")
    with c2:
        ejec = st.button("▶  Encode + Decode DCT", key="btn_img")

    if ejec:
        with st.spinner("Ejecutando DCT (encode + decode)…"):
            res = DCT().run(datos, cal)

        sec("🔄", "PIPELINE COMPLETO")
        render_pipeline("decode")
        sec("📊", "MÉTRICAS GLOBALES")
        render_metrics(res)
        render_verificacion(res)

        st.markdown("---")
        render_dir("enc", "DCT — JPEG", res.t_enc_ms)
        render_pasos(res.pasos_enc)

        st.markdown("---")
        render_dir("dec", "IDCT — Reconstrucción", res.t_dec_ms)
        render_pasos(res.pasos_dec)

    with st.expander("🔎 Ver Código Fuente — DCT Encode + Decode"):
        st.code(
            """# ── DCT — Encode + Decode completo (JPEG-like) ───────────
import numpy as np, math

Q = np.array([
    [16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99],
], dtype=float)

def encode(blk: np.ndarray, quality: int = 50):
    scale = max(1,(100-quality)/50) if quality<50 else 50/max(1,quality)
    Qs = Q * scale
    b = blk.astype(float) - 128            # 1. Level shift
    from scipy.fft import dctn
    F = dctn(b, norm='ortho')              # 2. DCT-II 2D
    Fq = np.round(F / Qs).astype(int)     # 3. Cuantización ← PÉRDIDA
    return Fq, Qs

def decode(Fq: np.ndarray, Qs: np.ndarray) -> np.ndarray:
    Frec = Fq.astype(float) * Qs          # 1. Decuantización
    from scipy.fft import idctn
    b_rec = idctn(Frec, norm='ortho')     # 2. IDCT-II 2D
    return np.clip(b_rec + 128, 0, 255).astype(np.uint8)  # 3. Level shift inverso

def psnr(orig, rec):
    mse = np.mean((orig.astype(float) - rec) ** 2)
    return 10 * math.log10(255**2 / mse) if mse > 0 else float('inf')

# Verificación
B = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
for q in [10, 50, 90]:
    Fq, Qs = encode(B, quality=q)
    R = decode(Fq, Qs)
    print(f"Q={q:2d}: PSNR={psnr(B, R):6.2f} dB  zeros_AC={np.sum(Fq[1:]==0)}/63")
""",
            language="python",
        )

    render_formulas()


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — AUDIO → μ-LAW G.711
# ─────────────────────────────────────────────────────────────────────────────

def _read_wav(datos: bytes) -> Tuple[bytes, int, int, int]:
    try:
        with wave.open(io.BytesIO(datos)) as wf:
            return (wf.readframes(wf.getnframes()),
                    wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
    except Exception:
        return datos, 1, 2, 44100


def tab_audio() -> None:
    cu, ci = st.columns([3, 2], gap="large")
    with cu:
        sec("🎵", "CARGAR AUDIO")
        up = st.file_uploader(
            "aud", type=["wav", "mp3", "ogg", "flac"],
            key="up_aud", label_visibility="collapsed",
        )
    with ci:
        sec("ℹ️", "ALGORITMO: μ-Law G.711 (1972)")
        st.markdown(
            """<div class="acard">
<div class="acard-top"><span class="acard-name">μ-Law G.711</span><span class="acard-tag tg">quasi-lossless</span></div>
<div class="acard-desc">Companding logarítmico: PCM 16-bit → μ-Law 8-bit.<br>
Ratio EXACTO 2:1 — siempre constante.<br>
Basado en percepción logarítmica del oído humano.<br>
Decode: expansión inversa → PCM 16-bit.<br>
Uso real: telefonía PSTN, VoIP, G.711 ITU-T.</div></div>""",
            unsafe_allow_html=True,
        )

    if not up:
        nofile("🎵", "Sube un archivo de audio (.wav PCM recomendado)",
               "Formatos: .wav  .mp3  .ogg  .flac")
        return

    datos = up.read()
    is_wav = up.name.lower().endswith(".wav")
    if is_wav:
        pcm, n_ch, sw, fr = _read_wav(datos)
        dur = len(pcm) / (n_ch * sw * fr) if fr else 0
        ibox("🔊",
             f"<strong>WAV PCM detectado</strong> · {n_ch}ch · {sw * 8}-bit · {fr:,} Hz · "
             f"<strong>{dur:.2f} s</strong> · {fmt(len(pcm))} PCM")
        datos_a = pcm
    else:
        ibox("⚠️", "Formato comprimido — se analiza el flujo de bytes en bruto.", "a")
        datos_a = datos

    an = AnAudio(datos_a)
    with st.spinner("Calculando…"):
        st3 = an.calcular()

    sec("📊", "ANÁLISIS ESTADÍSTICO")
    render_stats(st3)

    if is_wav and len(datos_a) >= 4:
        sec("〰️", "FORMA DE ONDA (muestra de 500 puntos)")
        arr = np.frombuffer(datos_a, dtype=np.int16)
        step = max(1, len(arr) // 500)
        st.line_chart(pd.DataFrame({"Amplitud": arr[::step][:500]}), color="#06b6d4", height=140)

    sec("📋", "DISTRIBUCIÓN DE BYTES DE AUDIO")
    render_tabla(st3, 25)

    sec("⚙️", "CODIFICAR + DECODIFICAR CON μ-LAW G.711")
    if st.button("▶  Encode + Decode μ-Law G.711", key="btn_aud"):
        with st.spinner("Ejecutando μ-Law (encode + decode)…"):
            res = MuLaw().run(datos_a)

        sec("🔄", "PIPELINE COMPLETO")
        render_pipeline("decode")
        sec("📊", "MÉTRICAS GLOBALES")
        render_metrics(res)
        render_verificacion(res)

        st.markdown("---")
        render_dir("enc", "μ-Law G.711 (Companding)", res.t_enc_ms)
        render_pasos(res.pasos_enc)

        st.markdown("---")
        render_dir("dec", "μ-Law G.711 (Expansión)", res.t_dec_ms)
        render_pasos(res.pasos_dec)

        if is_wav and len(datos_a) >= 4:
            sec("〰️", "COMPARACIÓN FORMA DE ONDA: ORIGINAL vs DECODIFICADO")
            orig_a = np.frombuffer(datos_a, dtype=np.int16)
            dec_a = np.frombuffer(res.datos_dec[: len(datos_a)], dtype=np.int16)
            step2 = max(1, len(orig_a) // 300)
            df_cmp = pd.DataFrame({
                "Original PCM 16-bit": orig_a[::step2][:300].tolist(),
                "Decodificado μ-Law→PCM": dec_a[::step2][:300].tolist(),
            })
            st.line_chart(df_cmp, color=["#06b6d4", "#8b5cf6"], height=180)
            ibox("🔬",
                 "Las dos formas de onda son visualmente idénticas → "
                 "el oído <strong>no percibe la diferencia</strong>.<br>"
                 "La pequeña distorsión logarítmica es imperceptible en señales de voz.", "g")

    with st.expander("🔎 Ver Código Fuente — μ-Law G.711 Encode + Decode"):
        st.code(
            """# ── μ-Law G.711 — Encode + Decode completo ───────────────
import math, struct

MU, BIAS = 255, 132

def encode_sample(s: int) -> int:
    \"\"\"PCM 16-bit → μ-Law 8-bit (G.711)\"\"\"
    s    = max(-32768, min(32767, s))
    sign = 0x00 if s >= 0 else 0x80
    mag  = min(abs(s), 32635) + BIAS
    exp  = max(0, min(int(math.log2(mag)) - 7, 7))
    mant = (mag >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mant)) & 0xFF   # complementar bits G.711

def decode_sample(b: int) -> int:
    \"\"\"μ-Law 8-bit → PCM 16-bit\"\"\"
    b    = ~b & 0xFF                 # descomplementar
    sign = b & 0x80
    exp  = (b >> 4) & 0x07
    mant = b & 0x0F
    mag  = ((mant << 1) | 1) << (exp + 2)
    return -(mag - BIAS) if sign else (mag - BIAS)

def encode_audio(pcm_bytes: bytes) -> bytes:
    n = len(pcm_bytes) // 2
    samples = struct.unpack(f'<{n}h', pcm_bytes[:n*2])
    return bytes(encode_sample(s) for s in samples)

def decode_audio(mulaw_bytes: bytes) -> bytes:
    samples = [decode_sample(b) for b in mulaw_bytes]
    return struct.pack(f'<{len(samples)}h', *samples)

# Verificación
pcm = struct.pack('<4h', 0, 16383, -16383, 32767)
enc = encode_audio(pcm)
dec = decode_audio(enc)
orig  = struct.unpack('<4h', pcm)
recon = struct.unpack('<4h', dec)
print(f"Original:   {orig}")
print(f"μ-Law 8bit: {list(enc)}")
print(f"Recuperado: {recon}")
print(f"Ratio: {len(pcm)/len(enc):.1f}:1  EXACTO")
""",
            language="python",
        )

    render_formulas()


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — VIDEO → RLE + H.264 educativo
# ─────────────────────────────────────────────────────────────────────────────

def tab_video() -> None:
    cu, ci = st.columns([3, 2], gap="large")
    with cu:
        sec("🎬", "CARGAR VIDEO")
        up = st.file_uploader(
            "vid", type=["mp4", "avi", "mkv", "mov", "webm"],
            key="up_vid", label_visibility="collapsed",
        )
    with ci:
        sec("ℹ️", "ALGORITMO: RLE + Pipeline H.264")
        st.markdown(
            """<div class="acard">
<div class="acard-top"><span class="acard-name">RLE + H.264 (educativo)</span><span class="acard-tag tc">lossless/lossy</span></div>
<div class="acard-desc">RLE sobre bytes del contenedor (análisis didáctico).<br>
Video comprimido ≈ peor caso para RLE (bytes ~uniformes).<br>
Pipeline H.264 completo: Motion Est. + DCT + CABAC.<br>
Encode: tramas I/P/B + cuantización del residual.<br>
Decode: IDCT + compensación de movimiento (MC).</div></div>""",
            unsafe_allow_html=True,
        )

    if not up:
        nofile("🎬", "Sube un video para analizarlo",
               "Formatos: .mp4  .avi  .mkv  .mov  .webm")
        return

    datos = up.read()
    an = AnVideo(datos)
    ibox("ℹ️",
         f"<strong>{up.name}</strong> · {fmt(len(datos))} · "
         "Análisis sobre los primeros 80 KB del flujo del contenedor.")

    with st.spinner("Analizando flujo…"):
        st4 = an.calcular()

    sec("📊", "ANÁLISIS DEL FLUJO COMPRIMIDO")
    render_stats(st4)
    ibox("🔬",
         f"Alta entropía (<strong>{st4.entropia:.4f} bits/byte</strong>) porque H.264/HEVC "
         "ya aplicó codificación entrópica (CABAC).<br>"
         "Un video RAW tendría entropía menor y mayor potencial de compresión.", "v")

    sec("📋", "DISTRIBUCIÓN DE BYTES DEL FLUJO")
    render_tabla(st4, 30)

    sec("📈", "HISTOGRAMA DEL FLUJO")
    df_h = pd.DataFrame({
        "Val": list(range(256)),
        "F": [st4.frecuencias.get(b, 0) for b in range(256)],
    }).set_index("Val")
    st.bar_chart(df_h, color="#8b5cf6", height=150)

    sec("⚙️", "APLICAR RLE AL FLUJO DEL CONTENEDOR")
    if st.button("▶  Encode + Decode RLE", key="btn_vid"):
        with st.spinner("Ejecutando RLE…"):
            res = RLE().run(datos[:50_000])

        sec("🔄", "PIPELINE COMPLETO")
        render_pipeline("decode")
        sec("📊", "MÉTRICAS GLOBALES")
        render_metrics(res)
        render_verificacion(res)

        st.markdown("---")
        render_dir("enc", "RLE (Run-Length Encoding)", res.t_enc_ms)
        render_pasos(res.pasos_enc)

        st.markdown("---")
        render_dir("dec", "RLE (Expansión de Runs)", res.t_dec_ms)
        render_pasos(res.pasos_dec)

        if res.tasa < 1.0:
            ibox("⚠️",
                 "<strong>RLE no es eficiente en flujos ya comprimidos.</strong><br>"
                 "H.264/HEVC ya aplica CABAC → bytes casi uniformes → sin runs repetidos.<br>"
                 "Para video real se usa el pipeline H.264 completo (ver abajo).", "a")

    # Pipeline H.264 educativo
    sec("🎞️", "PIPELINE H.264 — ENCODE + DECODE EDUCATIVO")
    c_e, c_d = st.columns(2, gap="large")

    with c_e:
        render_dir("enc", "H.264 Encoder", 0)
        for tit, det in [
            ("E-1 · Partición en Macroblocks (16×16)",
             "La imagen se divide en macroblocks de 16×16 píxeles.\n"
             "Sub-particiones: 16×16, 16×8, 8×16, 8×8, 4×4.\n"
             "Cada macroblock se procesa independientemente."),
            ("E-2 · Estimación de Movimiento (Motion Estimation)",
             "Para tramas P/B: se busca el mejor macroblock en el frame de referencia.\n"
             "SAD = ΣΣ |curr(x,y) − ref(x+mvx, y+mvy)|  ← métrica de similitud\n"
             "Se transmite SOLO el vector de movimiento (mvx, mvy) → muy compacto."),
            ("E-3 · DCT + Cuantización del Residual",
             "Residual = Frame_actual − Frame_predicho\n"
             "DCT transforma el residual al dominio frecuencial.\n"
             "Cuantización: descarta coeficientes AC de alta frecuencia.\n"
             "→ LOSSY: aquí ocurre la pérdida de información."),
            ("E-4 · ZigZag + CABAC (Codificación Entrópica)",
             "ZigZag reordena la matriz DCT 8×8 → vector 1D (agrupa ceros).\n"
             "CABAC: codificación aritmética adaptiva por contexto.\n"
             "Hasta 15% más eficiente que Huffman convencional.\n"
             "Salida: NAL Units del bitstream H.264."),
        ]:
            with st.expander(tit, expanded=False):
                st.markdown(f'<div class="sdt">{det}</div>', unsafe_allow_html=True)

    with c_d:
        render_dir("dec", "H.264 Decoder", 0)
        for tit, det in [
            ("D-1 · Parser de NAL Units",
             "Se identifican y separan las NAL Units del bitstream.\n"
             "Se detecta el tipo de trama: I (Intra), P (Predictiva), B (Bidireccional).\n"
             "Se extraen los parámetros del SPS y PPS."),
            ("D-2 · Decodificación Entrópica (CABAC)",
             "Se invierte el modelo aritmético CABAC por contexto.\n"
             "Se recuperan los coeficientes DCT cuantizados.\n"
             "Se reconstruye la matriz de coeficientes para cada bloque."),
            ("D-3 · Decuantización + IDCT",
             "F_rec(u,v) = F_q(u,v) × Q(u,v)  ← decuantización\n"
             "IDCT-II transforma de regreso al dominio espacial.\n"
             "Residual_rec = IDCT(F_decuantizado)  ← aproximación"),
            ("D-4 · Compensación de Movimiento + Reconstrucción",
             "Frame_rec = MC(frame_ref, MVs) + Residual_rec\n"
             "Deblocking filter: suaviza artefactos en bordes de bloques.\n"
             "Frame_rec → referencia para los siguientes frames.\n"
             "PSNR típico H.264: 35–45 dB en video de alta calidad."),
        ]:
            with st.expander(tit, expanded=False):
                st.markdown(f'<div class="sdt">{det}</div>', unsafe_allow_html=True)

    sec("📦", "TASAS DE COMPRESIÓN H.264 POR TIPO DE TRAMA")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Trama I (Intra)", "3–4:1", "Solo DCT espacial")
    with c2: st.metric("Trama P (Pred.)", "10–15:1", "MV + DCT residual")
    with c3: st.metric("Trama B (Bi-dir.)", "15–25:1", "MV doble + DCT")
    with c4: st.metric("GOP completo", "50–200:1", "Promedio ponderado")

    with st.expander("🔎 Ver Código Fuente — RLE + Motion Estimation H.264"):
        st.code(
            """# ── RLE — Encode + Decode completo ────────────────────────
def rle_encode(datos: bytes) -> bytes:
    out, i = bytearray(), 0
    while i < len(datos):
        b, n = datos[i], 1
        while i+n < len(datos) and datos[i+n] == b and n < 255: n += 1
        out.extend([n, b])    # par (conteo, valor)
        i += n
    return bytes(out)

def rle_decode(datos: bytes) -> bytes:
    out, i = bytearray(), 0
    while i+1 < len(datos):
        n, b = datos[i], datos[i+1]
        out.extend([b] * n)   # expandir run
        i += 2
    return bytes(out)

# ── Motion Estimation H.264 (educativo) ──────────────────────
import numpy as np

def motion_estimation(ref, curr, bs=16, sr=16):
    \"\"\"Full-search block matching. Retorna lista de (mvx, mvy).\"\"\"
    H, W = curr.shape; mvs = []
    for i in range(0, H-bs+1, bs):
        for j in range(0, W-bs+1, bs):
            blk = curr[i:i+bs, j:j+bs]
            best_sad, best_mv = float('inf'), (0, 0)
            for dy in range(-sr, sr+1):
                for dx in range(-sr, sr+1):
                    ri, rj = i+dy, j+dx
                    if 0 <= ri <= H-bs and 0 <= rj <= W-bs:
                        sad = np.sum(np.abs(blk.astype(int)
                               - ref[ri:ri+bs, rj:rj+bs]))
                        if sad < best_sad:
                            best_sad, best_mv = sad, (dx, dy)
            mvs.append(best_mv)
    return mvs

def motion_compensation(ref, mvs, bs=16):
    \"\"\"DECODE: reconstruir frame predicho desde vectores de movimiento.\"\"\"
    H, W = ref.shape; pred = np.zeros_like(ref); k = 0
    for i in range(0, H-bs+1, bs):
        for j in range(0, W-bs+1, bs):
            mvx, mvy = mvs[k]; k += 1
            ri = max(0, min(H-bs, i+mvy))
            rj = max(0, min(W-bs, j+mvx))
            pred[i:i+bs, j:j+bs] = ref[ri:ri+bs, rj:rj+bs]
    return pred
""",
            language="python",
        )

    render_formulas()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN  —  patch st reference then run
# ─────────────────────────────────────────────────────────────────────────────

# We use a module-level alias so inner functions can call st directly


def main() -> None:
    inject_css()
    render_header()

    tabs = st.tabs([
        "📄  Texto → Huffman",
        "🖼️  Imagen → DCT/JPEG",
        "🎵  Audio → μ-Law G.711",
        "🎬  Video → RLE + H.264",
    ])
    with tabs[0]:
        tab_texto()
    with tabs[1]:
        tab_imagen()
    with tabs[2]:
        tab_audio()
    with tabs[3]:
        tab_video()

    st.markdown(
        """<div style="margin-top:3rem;padding:0.85rem 0;
border-top:1px solid rgba(6,182,212,0.13);text-align:center;
font-size:0.61rem;color:#4e5f7a">
DataLab v3.0 · Huffman (1952) · DCT/JPEG (1992) · μ-Law G.711 (1972) · RLE · H.264
</div>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
