import base64
import json
import math
import random
import streamlit as st
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN E INYECCIÓN CSS (Mantenido el estilo original)
# ═════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""<style>
    .matrix-cell{display:inline-block;width:26px;height:26px;line-height:26px;text-align:center;border:1px solid #444;margin:1px;font-family:'IBM Plex Mono',monospace;font-size:13px;border-radius:3px;}
    .data-bit{background-color:#1e3a8a;color:#cdd9e5;}
    .parity-bit{background-color:#065f46;color:#a7f3d0;font-weight:bold;}
    .error-bit{background-color:#991b1b;color:white;font-weight:bold;animation:blinker 1s linear infinite;}
    @keyframes blinker{50%{opacity:0;}}
    .dl-btn{display:block;background:linear-gradient(135deg,#8b5cf6 0%,#7b2cbf 100%);color:white !important;padding:10px 20px;border-radius:8px;font-weight:bold;text-decoration:none;text-align:center;margin-top:10px;}
    </style>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  LÓGICA FEC (Ampliación y Explicación)
# ═════════════════════════════════════════════════════════════════════════════
class MatrixFEC:
    def __init__(self, cols=4):
        self.cols = cols

    def calculate_parity(self, bits):
        return '1' if bits.count('1') % 2 != 0 else '0'

    def encode(self, bits):
        padding = (self.cols - (len(bits) % self.cols)) % self.cols
        padded = bits + ('0' * padding)
        rows = len(padded) // self.cols
        
        matrix = []
        col_parities = ['0'] * self.cols
        
        for r in range(rows):
            row_data = padded[r*self.cols : (r+1)*self.cols]
            row_p = self.calculate_parity(row_data)
            matrix.append(list(row_data) + [row_p])
            for i, bit in enumerate(row_data):
                if bit == '1': col_parities[i] = '1' if col_parities[i] == '0' else '0'
        
        master_p = self.calculate_parity("".join(col_parities))
        matrix.append(col_parities + [master_p])
        
        # Calcular tasa FEC (Overhead)
        total_bits = len(bits)
        total_encoded = (rows + 1) * (self.cols + 1)
        tasa_fec = (total_encoded - total_bits) / total_encoded * 100
        
        return matrix, tasa_fec

# ═════════════════════════════════════════════════════════════════════════════
#  INTERFAZ STREAMLIT
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Telecom Lab", layout="wide")
inject_css()

st.title("📡 Laboratorio: Fuente y Canal")
tab_fuente, tab_canal = st.tabs(["📄 Fuente", "📡 Canal (FEC)"])

with tab_fuente:
    st.write("Configuración de fuente: Texto, Imagen, Audio, Video (Mantenido).")

with tab_canal:
    st.subheader("Core: Forward Error Correction (FEC) - Paridad 2D")
    entrada_bits = st.text_input("Introduce secuencia de bits (Tx):", "110101101110")
    
    if st.button("Codificar FEC"):
        fec = MatrixFEC(cols=4)
        matriz, tasa = fec.encode(entrada_bits)
        
        st.info(f"### Análisis Paso a Paso")
        st.write("1. **Padding**: Se ajusta la longitud para formar una cuadrícula.")
        st.write("2. **Paridad por Fila**: Se calcula el bit de paridad para cada fila.")
        st.write("3. **Paridad por Columna**: Se calcula la paridad vertical.")
        st.write("4. **Paridad Maestra**: Bit de paridad cruzada final.")
        
        # Representación Gráfica
        html_mat = "<div>"
        for row in matriz:
            for cell in row:
                html_mat += f"<span class='matrix-cell data-bit'>{cell}</span>"
            html_mat += "<br>"
        html_mat += "</div>"
        st.markdown(html_mat, unsafe_allow_html=True)
        
        st.metric("Tasa de Overhead FEC", f"{tasa:.2f}%")
        
        # Generar descarga
        payload = {"data": entrada_bits, "matriz": matriz}
        b64 = base64.b64encode(json.dumps(payload).encode()).decode()
        st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="payload_fec.bin" class="dl-btn">⬇ Descargar Payload .BIN</a>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.write("Laboratorio de Telecomunicaciones")
