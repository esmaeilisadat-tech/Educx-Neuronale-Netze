import nbformat

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

fortgeschrittene_formulas = {
    1: """### Mathematische Analyse und Auswertung (Exercise 1)

**1. Mathematische Problemstellung:**
Stabilisierung des Netzwerks durch Batch Normalisierung.

**2. Hauptformel (Batch Normalization):**
$$ \hat{x}_i = \\frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \quad \\text{und} \quad y_i = \gamma \cdot \hat{x}_i + \\beta $$

**Parameter-Definitionen:**
- $x_i$: Der rohe Pixelwert/Aktivierungswert in einem Kanal.
- $\mu_B$: Der Mittelwert des aktuellen Batches (für diesen Kanal).
- $\sigma_B^2$: Die Varianz des aktuellen Batches.
- $\epsilon$: Eine winzige Konstante, um Division durch Null zu verhindern.
- $\gamma$: Ein gelernter Skalierungsfaktor.
- $\\beta$: Ein gelernter Verschiebungsfaktor.
- $y_i$: Der finale, normalisierte Ausgabewert.

**Mathematische Einsetzung:**
In unserem Code ist die Batch-Größe $32$. Nehmen wir an, das Modell berechnet für Pixel $i$ den Wert $x_i = 10.0$. Der Mittelwert aller 32 Bilder an dieser Stelle sei $\mu_B = 6.0$ und die Varianz $\sigma_B^2 = 4.0$. Die Keras-Konstante ist $\epsilon = 0.001$.
Einsetzen:
$$ \hat{x}_i = \\frac{10.0 - 6.0}{\sqrt{4.0 + 0.001}} \\approx \\frac{4.0}{2.0} = 2.0 $$
Angenommen das Netz hat $\gamma = 1.5$ und $\\beta = -0.5$ gelernt:
$$ y_i = 1.5 \cdot 2.0 + (-0.5) = 2.5 $$
Anstatt eines riesigen Wertes $10.0$ wird ein stabilerer Wert $2.5$ an die nächste Schicht gegeben.

**3. Analyse von Input und Output:**
- **Input:** $32 \\times 32 \\times 3$ CIFAR-Bilder.
- **Output:** Ein stabiles Training. Extreme Ausreißer in $x_i$ werden mathematisch zentriert.""",

    2: """### Mathematische Analyse und Auswertung (Exercise 2)

**1. Mathematische Problemstellung:**
Künstliche Vergrößerung der Datensätze zur Vermeidung von Overfitting durch Rotations-Matrizen.

**2. Hauptformel (Rotationsmatrix):**
$$ \\begin{bmatrix} x' \\\\ y' \end{bmatrix} = \\begin{bmatrix} \cos \\theta & -\sin \\theta \\\\ \sin \\theta & \cos \\theta \end{bmatrix} \\begin{bmatrix} x \\\\ y \end{bmatrix} $$

**Parameter-Definitionen:**
- $x, y$: Die ursprüngliche Koordinate eines Pixels.
- $x', y'$: Die neue, rotierte Koordinate des Pixels.
- $\\theta$: Der zufällige Rotationswinkel.

**Mathematische Einsetzung:**
Im Code haben wir `RandomRotation(0.1)` definiert, was max. $10\%$ eines Vollkreises ($36^\circ$) erlaubt. Nehmen wir an, das Bild rotiert zufällig um $\\theta = 30^\circ$. In Bogenmaß: $\cos(30^\circ) \\approx 0.866$ und $\sin(30^\circ) = 0.5$.
Angenommen, ein Frosch-Auge liegt bei der Koordinate $x=10, y=5$.
$$ \\begin{bmatrix} x' \\\\ y' \end{bmatrix} = \\begin{bmatrix} 0.866 & -0.5 \\\\ 0.5 & 0.866 \end{bmatrix} \\begin{bmatrix} 10 \\\\ 5 \end{bmatrix} $$
$$ x' = (0.866 \cdot 10) - (0.5 \cdot 5) = 8.66 - 2.5 = 6.16 $$
$$ y' = (0.5 \cdot 10) + (0.866 \cdot 5) = 5.0 + 4.33 = 9.33 $$
Das Auge des Froschs wurde im Tensor von $(10, 5)$ auf $(\\approx 6, 9)$ verschoben.

**3. Analyse von Input und Output:**
- **Input:** Unveränderte Bilder ($x, y$).
- **Output:** Bilder, deren Pixel künstlich verschoben wurden ($x', y'$). Das Modell lernt nun das Objekt (Frosch) unabhängig von seiner Koordinate, was die Validierungs-Genauigkeit stabilisiert.""",

    3: """### Mathematische Analyse und Auswertung (Exercise 3)

**1. Mathematische Problemstellung:**
Reduzierung der räumlichen Dimensionen durch Average Pooling.

**2. Hauptformel (Average Pooling):**
$$ y(i,j) = \\frac{1}{S_h \cdot S_w} \sum_{m=0}^{S_h-1} \sum_{n=0}^{S_w-1} x(2i+m, 2j+n) $$

**Parameter-Definitionen:**
- $y(i,j)$: Der neue, zusammengefasste Ausgabewert.
- $x$: Das Eingangsbild bzw. die Eingangs-Merkmalskarte.
- $S_h, S_w$: Die Höhe und Breite des Pooling-Fensters.

**Mathematische Einsetzung:**
In unserem Code nutzen wir `AveragePooling2D((2, 2))`, also $S_h=2, S_w=2$. Der Vorfaktor ist somit $\\frac{1}{4}$.
Nehmen wir einen $2 \\times 2$ Block aus der Matrix:
$$ x_{Block} = \\begin{bmatrix} 2.0 & 4.0 \\\\ 0.0 & 6.0 \end{bmatrix} $$
Wir setzen ein:
$$ y = \\frac{1}{4} (2.0 + 4.0 + 0.0 + 6.0) $$
$$ y = \\frac{1}{4} (12.0) = 3.0 $$
Der gesamte $2 \\times 2$ Block wird zu einem einzigen Pixel mit dem Wert $3.0$ reduziert.

**3. Analyse von Input und Output:**
- **Input:** $2000$ Bilder und $7$ verschiedene Konfigurationen.
- **Output:** Average Pooling liefert homogenere Reduktionen als MaxPooling. Wie wir im Output-Streudiagramm sehen, beeinflussen solche Architekturentscheidungen die Parameteranzahl und Genauigkeit messbar."""
}

experte_formulas = {
    1: """### Mathematische Analyse und Auswertung (Exercise 1)

**1. Mathematische Problemstellung:**
Saliency Maps – Berechnung der Relevanz jedes Pixels für die Klassifikation.

**2. Hauptformel (Gradienten-Saliency):**
$$ M_{i, j} = \left| \\frac{\partial S_c}{\partial I_{i, j}} \\right| $$

**Parameter-Definitionen:**
- $M_{i, j}$: Der Saliency-Wert (Wichtigkeit) für das Pixel $(i, j)$.
- $| \cdot |$: Der mathematische Betrag.
- $\partial S_c$: Die partielle Änderung des Output-Scores für die korrekte Klasse $c$.
- $\partial I_{i, j}$: Die partielle Änderung des Farbwertes des Pixels an der Stelle $(i, j)$.

**Mathematische Einsetzung:**
Angenommen, der unnormalisierte Output für die Klasse "Ziffer 7" ist $S_7$. Das Pixel $(14, 14)$ gehört zum Strich der 7. Wenn wir diesen Pixel $I_{14,14}$ von $0.5$ auf $0.6$ erhöhen (eine Änderung $\Delta I = 0.1$) und sich der Score $S_7$ dadurch von $12.0$ auf $13.5$ erhöht ($\Delta S_7 = 1.5$), so approximieren wir:
$$ \\frac{\partial S_7}{\partial I_{14, 14}} \\approx \\frac{1.5}{0.1} = 15.0 $$
$$ M_{14, 14} = |15.0| = 15.0 $$
Ein Hintergrundpixel an $(0,0)$ ändert den Score kaum ($\Delta S \\approx 0$). Somit $M_{0,0} = 0$.

**3. Analyse von Input und Output:**
- **Input:** Pixel-Tensoren aus MNIST.
- **Output:** Das Heatmap-Bild visualisiert diese $M$-Werte. Da $M_{14,14}=15.0$ sehr hoch ist, wird dieses Pixel rot/gelb geplottet. Der Hintergrund ($M_{0,0}=0$) bleibt schwarz.""",

    2: """### Mathematische Analyse und Auswertung (Exercise 2)

**1. Mathematische Problemstellung:**
Reduzierung von Variablen durch Separierung der Multiplikationen (Depthwise Separable Convolution).

**2. Hauptformel (Parameter-Anzahl):**
$$ P_{std} = M^2 \cdot C_{in} \cdot C_{out} $$
$$ P_{sep} = (M^2 \cdot C_{in}) + (C_{in} \cdot C_{out}) $$

**Parameter-Definitionen:**
- $P_{std}$: Anzahl der Variablen (Parameter) bei Standard-Faltung.
- $P_{sep}$: Anzahl der Variablen bei getrennter (separable) Faltung.
- $M$: Größe des räumlichen Filters (z.B. $3$ für $3 \\times 3$).
- $C_{in}$: Anzahl der Eingabekanäle (z.B. Filter der Vorschicht).
- $C_{out}$: Anzahl der Ausgabekanäle.

**Mathematische Einsetzung:**
In unserem Code vergleichen wir Schichten. Nehmen wir an $M=3, C_{in}=32, C_{out}=64$.
Einsetzen in Standard-Faltung:
$$ P_{std} = 3^2 \cdot 32 \cdot 64 = 9 \cdot 2048 = 18.432 $$
Einsetzen in Separable Faltung:
$$ P_{sep} = (3^2 \cdot 32) + (32 \cdot 64) $$
$$ P_{sep} = 288 + 2048 = 2.336 $$

**3. Analyse von Input und Output:**
- **Input:** $32 \\times 32 \\times 3$ CIFAR-Bilder.
- **Output:** Im Plot sehen wir, dass $18.432$ auf $2.336$ reduziert wurde, eine massive Variablen-Ersparnis. Die Genauigkeit auf der Y-Achse beweist, dass $P_{sep}$ mathematisch fast denselben Output erzeugt wie $P_{std}$, aber die Trainingszeit und der Speicherbedarf drastisch fallen.""",

    3: """### Mathematische Analyse und Auswertung (Exercise 3)

**1. Mathematische Problemstellung:**
Simulation von stochastischem Sensorrauschen in künstlich generierten Bilddaten.

**2. Hauptformel (Additives Gauß-Rauschen):**
$$ I_{neu}(x, y) = I_{alt}(x, y) + z $$
mit der Wahrscheinlichkeitsdichte für $z$:
$$ p(z) = \\frac{1}{\sigma \sqrt{2\pi}} e^{-\\frac{1}{2}\left(\\frac{z-\mu}{\sigma}\\right)^2} $$

**Parameter-Definitionen:**
- $I_{neu}(x, y)$: Der endgültige verrauschte Pixelwert.
- $I_{alt}(x, y)$: Der ideale geometrische Pixelwert (ohne Rauschen).
- $z$: Eine Zufallsvariable, gezogen aus der Normalverteilung (Gauß-Kurve).
- $\mu$: Der Mittelwert des Rauschens (hier 0).
- $\sigma$: Die Standardabweichung des Rauschens (hier 0.05).
- $e$: Eulersche Zahl ($\\approx 2.718$).

**Mathematische Einsetzung:**
Für unsere grauen Hintergrundpixel ist $I_{alt} = 0.78$ ($200/255$). Wir ziehen eine Zufallszahl $z$ mit $\mu=0$ und $\sigma=0.05$.
Angenommen, der Zufallsgenerator liefert $z = -0.04$ für das Pixel $(5,5)$:
$$ I_{neu}(5, 5) = 0.78 + (-0.04) = 0.74 $$
Für das Pixel $(6,5)$ zieht er vielleicht $z = +0.08$:
$$ I_{neu}(6, 5) = 0.78 + 0.08 = 0.86 $$
Das ehemals flache Grau hat nun hoch- und niedrigwertige Flecken (Rauschen).

**3. Analyse von Input und Output:**
- **Input:** Perfekte geometrische Vektoren.
- **Output:** Das Modell erhält als Input Tensoren, bei denen jedes Pixel um $\sigma=0.05$ abweicht. Wie die Ergebnisseite beweist ($100\%$ Accuracy), sind die Faltungsfilter des CNNs iterativ robust gegen lokales Rauschen $z$ und erkennen erfolgreich die darunter liegende Form (Kreis, Quadrat)."""
}

p1 = r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_07_CNN\notebooks\Fortgeschrittene.ipynb'
p2 = r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_07_CNN\notebooks\Experte.ipynb'

rewrite_formulas(p1, fortgeschrittene_formulas)
rewrite_formulas(p2, experte_formulas)
