# 📋 PLANTILLA DE SUSTENTACIÓN
## DataLab · Compresión de Datos · Teoría de la Información
### Laboratorio Universitario

---

## ESTRUCTURA DE LA PRESENTACIÓN (45–60 min)

---

## 🔷 PARTE 1 — MARCO TEÓRICO (10 min)

### Diapositiva 1: Portada
- **Título**: Laboratorio de Compresión de Datos
- **Subtítulo**: Análisis basado en Teoría de la Información de Shannon (1948)
- Nombres, código, fecha, docente

---

### Diapositiva 2: Motivación
**Pregunta de apertura al jurado:**
> "¿Por qué una imagen PNG de 5 MB puede pesar 500 KB como JPEG? ¿Qué información se descarta y cómo se mide?"

**Contexto:**
- En 2023, la humanidad generó ~120 zettabytes de datos
- Sin compresión, Netflix requeriría 4× más ancho de banda
- Toda compresión eficiente está fundamentada en la Teoría de la Información

---

### Diapositiva 3: Teoría de la Información — Shannon (1948)
**Definición de Entropía:**

```
H(X) = −∑ p(xᵢ) · log₂ p(xᵢ)    [bits/símbolo]
```

**Interpretación intuitiva:**
- H = 0 bits → datos perfectamente predecibles (sin información útil)
- H = log₂(N) bits → distribución uniforme (máxima incertidumbre)
- La entropía es el **límite teórico** de compresión sin pérdida

**Ejemplo en vivo (mostrar en la app):**
→ Subir `texto_ejemplo.txt` y mostrar H(X)

---

### Diapositiva 4: Métricas Fundamentales

| Métrica | Fórmula | Significado |
|---------|---------|-------------|
| Entropía H(X) | −∑ pᵢ·log₂(pᵢ) | Información promedio por símbolo |
| Long. Promedio L̄ | ∑ pᵢ·lᵢ | Bits por símbolo en el código real |
| Eficiencia η | H(X) / L̄ | Qué tan cerca estamos del óptimo |
| Redundancia R | 1 − η | Fracción de bits "desperdiciados" |

**Teorema de Shannon (Primer Teorema de Codificación):**
```
H(X) ≤ L̄_Huffman < H(X) + 1
```
→ Huffman siempre está dentro de 1 bit del óptimo.

---

## 🔷 PARTE 2 — ALGORITMOS (20 min)

### Diapositiva 5: Huffman — Texto

**Concepto central:** Asignar códigos cortos a símbolos frecuentes

**Procedimiento (mostrar en la app):**
1. Calcular frecuencias y probabilidades de cada carácter
2. Insertar todos los nodos en un min-heap
3. Fusionar los 2 nodos de menor frecuencia en un nodo padre
4. Repetir hasta tener un solo árbol
5. Asignar '0' a la rama izquierda y '1' a la derecha
6. El código de cada símbolo = camino desde la raíz hasta la hoja

**Demostración en vivo:**
→ Subir archivo `.txt`, seleccionar Huffman, ejecutar
→ Mostrar tabla de códigos y árbol
→ Señalar: η ≥ 90% en texto natural típico

**Pregunta frecuente del jurado:**
- *"¿Por qué Huffman no es óptimo para cualquier fuente?"*
  - Respuesta: es óptimo por símbolo. Para distribuciones donde H(X) es fraccionario (ej. H=1.3), L̄ puede ser hasta 1 bit mayor. Solución: codificación aritmética.

---

### Diapositiva 6: LZW — Texto/Binario

**Concepto central:** Diccionario adaptivo sin transmitirlo

**Diferencia clave vs Huffman:**
- Huffman: trabaja símbolo a símbolo
- LZW: trabaja con **cadenas de símbolos** → captura redundancia a mayor escala

**Procedimiento:**
1. Diccionario inicial: 256 entradas (ASCII)
2. Se extiende el buffer w mientras wc esté en el diccionario
3. Al no encontrar wc: emitir código de w, agregar wc al diccionario
4. El receptor reconstruye el mismo diccionario de forma autónoma

**Demostración:**
→ Mostrar tabla LZW de primeras 10 entradas
→ Señalar cómo el diccionario crece con patrones del texto

---

### Diapositiva 7: RLE — Imagen (BMP)

**Concepto central:** Explotar redundancia espacial

**Cuándo funciona bien vs mal:**

| Caso | Datos | Salida RLE | Ratio |
|------|-------|-----------|-------|
| Óptimo | `AAAAAAAAAA` (10) | `[10, A]` (2) | 5:1 |
| Neutro | `AAABBBCCC` (9) | `[3,A][3,B][3,C]` (6) | 1.5:1 |
| Peor caso | `ABCDEFGH` (8) | `[1,A][1,B]...` (16) | 0.5:1 |

**Demostración:**
→ Subir imagen BMP de colores planos vs foto → comparar ratio

---

### Diapositiva 8: DCT — Imagen JPEG

**Concepto central:** Transformación al dominio frecuencial para separar lo importante de lo descartable

**Pipeline JPEG:**
```
Imagen → Bloques 8×8 → Level Shift → DCT 2D → Cuantización → ZigZag → RLE + Huffman
```

**Propiedad clave del DCT:**
> Concentra la energía de la imagen en pocos coeficientes de baja frecuencia.
> Los coeficientes de alta frecuencia (detalles finos) pueden descartarse con mínima degradación perceptual.

**Ecuación DCT-II 2D:**
```
F(u,v) = (2/N)·Cᵤ·Cᵥ · ΣΣ f(x,y) · cos[(2x+1)uπ/2N] · cos[(2y+1)vπ/2N]
```

**Demostración:**
→ Subir imagen, aplicar DCT con calidad 10 vs 90
→ Mostrar: ratio de compresión vs calidad visual
→ Señalar coeficiente DC vs coeficientes AC en la tabla

---

### Diapositiva 9: μ-Law G.711 — Audio

**Concepto central:** El oído humano percibe la amplitud de forma logarítmica

**Motivación:**
- PCM lineal: misma resolución para señales fuertes y débiles
- μ-law: más resolución para señales débiles (donde el oído es más sensible)
- Resultado: 16-bit → 8-bit con calidad perceptual casi igual

**Fórmula:**
```
y = sgn(x) · ln(1 + μ|x|) / ln(1 + μ)    donde μ = 255
```

**Demostración:**
→ Subir `.wav`, aplicar μ-Law, mostrar ratio 2:1 exacto
→ Mostrar tabla de muestras PCM → μ-law

---

### Diapositiva 10: Video — Compresión Temporal

**Concepto central:** Explotar la redundancia entre frames consecutivos

**Tipos de tramas:**
- **Trama I (Intra)**: codificada solo con DCT espacial (~3.5:1)
- **Trama P (Predictiva)**: Motion Vector + DCT del residual (~12:1)
- **Trama B (Bidireccional)**: referencia frames anteriores Y futuros (~20:1)

**GOP (Group Of Pictures):**
```
I B B P B B P B B I B B P ...
↑                   ↑
Punto de acceso aleatorio
```

---

## 🔷 PARTE 3 — DEMOSTRACIÓN EN VIVO (15 min)

### Guion de demostración:

**Texto (5 min):**
1. Subir `lorem_ipsum.txt` (o código fuente Python)
2. Mostrar dashboard: H(X), L̄, η
3. Señalar los caracteres más frecuentes (espacios, 'e', 'a')
4. Ejecutar Huffman → mostrar tabla de códigos
5. Comparar: código del espacio (1-2 bits) vs 'z' (7-8 bits)
6. Mostrar código fuente del algoritmo en la app

**Imagen (5 min):**
1. Subir imagen PNG alta resolución
2. Mostrar histograma de bytes
3. Aplicar DCT Q=10 → señalar ratio alto
4. Aplicar DCT Q=90 → señalar ratio bajo pero calidad alta
5. Comparar con RLE en imagen de colores planos

**Audio (5 min):**
1. Subir `.wav` de voz
2. Mostrar forma de onda
3. Aplicar μ-Law → demostrar ratio exacto 2:1
4. Explicar por qué G.711 es el estándar telefónico mundial

---

## 🔷 PARTE 4 — ANÁLISIS Y CONCLUSIONES (5 min)

### Diapositiva 11: Comparación de Algoritmos

| Algoritmo | Tipo | Ratio típico | Complejidad | Aplicación |
|-----------|------|-------------|-------------|------------|
| Huffman | Lossless | 1.5–4:1 | O(n log n) | ZIP, PDF, PNG |
| LZW | Lossless | 2–5:1 | O(n) | GIF, TIFF, PDF |
| RLE | Lossless | Variable | O(n) | BMP, FAX |
| DCT-JPEG | Lossy | 10–50:1 | O(n log n) | JPEG, MPEG |
| μ-Law G.711 | Lossy* | 2:1 exacto | O(n) | VoIP, PSTN |
| ADPCM | Lossy* | 4:1 | O(n) | WAV, multimedia |
| H.264 | Lossy | 50–200:1 | O(n²) | Video streaming |

*Quasi-lossless: distorsión perceptualmente imperceptible

---

### Diapositiva 12: Conclusiones

1. **La Teoría de la Información provee los límites** — ningún compresor puede superar la entropía H(X) de forma sin pérdida

2. **Cada dominio tiene su redundancia característica:**
   - Texto → repetición de caracteres y palabras
   - Imagen → redundancia espacial entre píxeles vecinos
   - Audio → redundancia temporal + limitaciones del oído humano
   - Video → redundancia temporal entre frames

3. **Trade-off fundamental:** ratio de compresión ↔ calidad ↔ complejidad computacional

4. **La app implementa el pipeline completo** de análisis estadístico, con la arquitectura preparada para backends de compresión reales

---

## 🔷 PREGUNTAS FRECUENTES DEL JURADO

**Q1: ¿Puede un compresor superar la entropía de Shannon?**
> No, para datos sin pérdida. El Primer Teorema de Codificación de Shannon establece que H(X) es el límite inferior de la longitud promedio de código. Un compresor que "supere" H(X) sería incorrecto (no podría descomprimir correctamente).

**Q2: ¿Por qué Huffman no es óptimo en todos los casos?**
> Huffman es óptimo entre los códigos de longitud entera por símbolo. Si H(X) = 1.2 bits, el mejor código entero tiene L̄ = 2 bits (η = 60%). La codificación aritmética puede acercarse a H(X) = 1.2 bits arbitrariamente.

**Q3: ¿Qué es la redundancia y cómo se mide?**
> R = 1 - η = 1 - H(X)/L̄. Representa la fracción de bits que podrían eliminarse sin perder información. En inglés natural, R ≈ 50–75% (el texto tiene mucha redundancia).

**Q4: ¿Por qué JPEG usa bloques de 8×8 y no de otro tamaño?**
> Es un equilibrio entre localidad (bloques pequeños) y eficiencia de la DCT. 8×8 = 64 coeficientes, número razonable para cuantización. Bloques mayores capturarían más correlación pero con mayor complejidad O(N² log N).

**Q5: ¿Qué diferencia hay entre compresión con y sin pérdida?**
> Sin pérdida (lossless): la descompresión es idéntica al original. Bit a bit igual.
> Con pérdida (lossy): la descompresión aproxima el original. Se acepta distorsión medida por PSNR o SSIM. Solo viable cuando el receptor es imperfecto (el ojo, el oído).

**Q6: ¿Por qué la entropía del video comprimido (MP4) es alta?**
> Porque H.264/HEVC ya aplicó codificación entrópica (CABAC). Los bytes del contenedor MP4 tienen distribución casi uniforme → H(X) ≈ 7.5–8.0 bits/byte. Comprimir un MP4 con ZIP no reduce su tamaño por esto mismo.

**Q7: ¿Cómo escala la arquitectura OOP de la app a un compresor real?**
> Cada clase `Codificador*` tiene métodos bien definidos y `ResultadoCompresion` como interfaz de salida. Para un backend real solo se reemplaza el cuerpo de las funciones marcadas `STUB` sin cambiar la UI. Esto sigue el principio Open/Closed (SOLID).

---

## MATERIAL DE APOYO

### Archivos de prueba recomendados

```
Texto:   lorem_ipsum.txt (500+ palabras), código fuente Python
Imagen:  foto.jpg (≥1 MB), captura de pantalla (PNG con áreas sólidas)
Audio:   grabación_voz.wav (PCM 16-bit, 44100 Hz, mono/stereo)
Video:   clip_corto.mp4 (cualquier video ≤50 MB)
```

### Referencias bibliográficas

1. Shannon, C. E. (1948). *A Mathematical Theory of Communication*. Bell System Technical Journal.
2. Huffman, D. A. (1952). *A Method for the Construction of Minimum-Redundancy Codes*. Proceedings of the IRE.
3. Welch, T. A. (1984). *A Technique for High-Performance Data Compression*. IEEE Computer.
4. Wallace, G. K. (1992). *The JPEG Still Picture Compression Standard*. IEEE Transactions on Consumer Electronics.
5. ITU-T G.711 (1988). *Pulse Code Modulation of Voice Frequencies*.
6. Sayood, K. (2017). *Introduction to Data Compression* (5th ed.). Morgan Kaufmann.

---

*DataLab v1.0.0 · Laboratorio de Teoría de la Información*
