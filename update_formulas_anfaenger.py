import nbformat
import os

def rewrite_formulas(filepath, new_formulas):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    f_idx = 1
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            if '### Mathematische Analyse und Auswertung' in cell.source:
                if f_idx in new_formulas:
                    cell.source = new_formulas[f_idx]
                f_idx += 1

    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

anfaenger_formulas = {
    1: """### Mathematische Analyse und Auswertung (Exercise 1)

**1. Mathematische Problemstellung:**
Wir trainieren ein neuronales Netz zur Bildklassifikation.

**2. Hauptformel (Softmax-Ausgabe für Wahrscheinlichkeiten):**
$$ \hat{y}_c = \\frac{e^{z_c}}{\sum_{j=1}^{K} e^{z_j}} $$

**Parameter-Definitionen:**
- $\hat{y}_c$: Die berechnete Wahrscheinlichkeit, dass das Bild zur Klasse $c$ gehört.
- $z_c$: Der rohe Ausgabewert (Logit) des Netzwerks für die Klasse $c$.
- $K$: Die totale Anzahl der möglichen Klassen.
- $e$: Die Eulersche Zahl ($\\approx 2.718$).

**Mathematische Einsetzung (Beispiel für Ziffer 2):**
In unserem Code haben wir $K=10$ Klassen (Ziffern 0-9). Angenommen, das Netzwerk berechnet für ein Bild die Logits $z = [1.2, 0.1, 5.0, -1.0, \dots]$. Wir berechnen die Wahrscheinlichkeit für die Klasse $c=2$ (die dritte Klasse, Index 2):
$$ \hat{y}_2 = \\frac{e^{5.0}}{e^{1.2} + e^{0.1} + e^{5.0} + e^{-1.0} + \dots} $$
$$ \hat{y}_2 \\approx \\frac{148.41}{3.32 + 1.10 + 148.41 + 0.36 + \dots} \\approx 0.96 $$
Das Modell ist sich zu $96\%$ sicher, dass es sich um eine 2 handelt.

**3. Analyse von Input und Output:**
- **Input:** $60.000$ Trainingsbilder (jeweils $28 \\times 28$ Pixel).
- **Output:** Nach 5 Epochen konvergiert die Verlustfunktion und liefert genaue Wahrscheinlichkeiten $\hat{y}$ (Genauigkeit $\\approx 98.5\%$).""",

    2: """### Mathematische Analyse und Auswertung (Exercise 2)

**1. Mathematische Problemstellung:**
Manuelle Merkmalsextraktion mittels 2D-Faltung (ohne Netz).

**2. Hauptformel (Diskrete 2D-Faltung):**
$$ G(x, y) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I(x - m, y - n) \cdot K(m, n) $$

**Parameter-Definitionen:**
- $G(x, y)$: Der neue Pixelwert im gefalteten Bild an der Position $(x, y)$.
- $I(x, y)$: Der ursprüngliche Pixelwert im Originalbild.
- $K(m, n)$: Der Wert in der Filtermatrix (Kernel) an Position $(m, n)$.
- $M, N$: Die Höhe und Breite des Filters.

**Mathematische Einsetzung (Beispiel Weichzeichner):**
In unserem Code nutzen wir $3 \\times 3$ Filter ($M=3, N=3$). Für den Weichzeichner haben wir alle Werte auf $\\frac{1}{9}$ gesetzt:
$$ K = \\begin{bmatrix} 0.11 & 0.11 & 0.11 \\\\ 0.11 & 0.11 & 0.11 \\\\ 0.11 & 0.11 & 0.11 \\end{bmatrix} $$
Wir betrachten ein $3 \\times 3$ Fenster des Bildes $I$, wo ein weißer Strich (Pixelwert 1.0) neben schwarzem Hintergrund (Pixelwert 0.0) liegt:
$$ I_{Fenster} = \\begin{bmatrix} 1.0 & 1.0 & 0.0 \\\\ 1.0 & 1.0 & 0.0 \\\\ 0.0 & 0.0 & 0.0 \\end{bmatrix} $$
Wir setzen ein und multiplizieren elementweise:
$$ G(1, 1) = (1.0 \cdot 0.11) + (1.0 \cdot 0.11) + \dots + (0.0 \cdot 0.11) $$
$$ G(1, 1) = 4 \cdot 0.11 + 5 \cdot 0.0 = 0.44 $$
Der harte weiße Rand ($1.0$) wurde auf ein Grau ($0.44$) geglättet.

**3. Analyse von Input und Output:**
- **Input:** $28 \\times 28$ Pixelmatrix der Ziffer 5.
- **Output:** Die gefalteten Bilder ($G$) zeigen deutlich, dass je nach Kernel $K$ entweder Kanten extrahiert oder Kontraste geglättet wurden.""",

    3: """### Mathematische Analyse und Auswertung (Exercise 3)

**1. Mathematische Problemstellung:**
Extraktion der internen "Feature Maps" eines Netzes.

**2. Hauptformel (Aktivierung einer Feature Map):**
$$ A_k(x, y) = \max(0, (I * W_k)(x, y) + b_k) $$

**Parameter-Definitionen:**
- $A_k(x, y)$: Der Aktivierungswert des $k$-ten Filters am Pixel $(x, y)$.
- $\max(0, \cdot)$: Die ReLU-Aktivierungsfunktion.
- $I$: Das Eingabebild.
- $W_k$: Die gelernte Gewichtsmatrix des $k$-ten Filters.
- $b_k$: Der gelernte Bias (Verschiebung) für den $k$-ten Filter.

**Mathematische Einsetzung (Beispiel Filter 1):**
In unserem Code haben wir $16$ Filter ($k=1 \dots 16$). Angenommen, das Modell hat für Filter 1 ($k=1$) folgende Parameter an Position $(x,y)$ berechnet:
Der Faltungswert $(I * W_1)(x, y)$ sei $-1.2$ (ein negatives Ergebnis, z.B. weil das gesuchte Muster nicht gefunden wurde). Der Bias $b_1$ sei $0.5$.
Wir setzen in die Formel ein:
$$ A_1(x, y) = \max(0, -1.2 + 0.5) $$
$$ A_1(x, y) = \max(0, -0.7) = 0 $$
Das Pixel in der Feature Map wird exakt $0$ (schwarz im Plot). Wenn der Faltungswert stattdessen $+3.0$ wäre:
$$ A_1(x, y) = \max(0, 3.0 + 0.5) = 3.5 $$
Das Pixel wird hell und signalisiert dem Netz eine starke Muster-Erkennung.

**3. Analyse von Input und Output:**
- **Input:** Das $28 \\times 28 \\times 1$ Testbild.
- **Output:** Ein $28 \\times 28 \\times 16$ Tensor. Die grafische Darstellung zeigt 16 verschiedene Feature Maps $A_1 \dots A_{16}$. Durch ReLU ist das berechnete Minimum (`min=0.000`) im Konsolen-Output logisch erklärt."""
}

filepath = r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_07_CNN\notebooks\Anfaenger.ipynb'
rewrite_formulas(filepath, anfaenger_formulas)
