import nbformat
import sys

def replace_formula_cells(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # find where to replace
    # We know that the cell before the next exercise or at the end is the formula cell from our previous run.
    # Actually, we can just look for cells that start with '### Mathematische Formel' and replace them.
    
    ex_idx = 1
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and '### Mathematische Formel' in cell.source:
            if ex_idx in replacements:
                cell.source = replacements[ex_idx]
                ex_idx += 1

    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

anfaenger_reps = {
    1: """### Mathematische Analyse und Auswertung (Exercise 1)

**1. Mathematische Problemstellung:**
Wir betrachten ein Klassifikationsproblem, bei dem ein Eingabebild $X \in \mathbb{R}^{28 \\times 28 \\times 1}$ auf einen Wahrscheinlichkeitsvektor $\hat{y} \in \mathbb{R}^{10}$ abgebildet werden soll. Das Ziel ist es, die Parameter $\\theta$ des Convolutional Neural Networks (CNN) so zu optimieren, dass die Vorhersage $\hat{y}$ möglichst genau mit dem wahren Label $y$ übereinstimmt.

**2. Implementierte mathematische Operationen:**
- **2D-Faltung (Convolution):** 
  Das Netzwerk verwendet Filter $K$, die über das Bild gleiten. Für einen Pixel an Position $(i,j)$ lautet die Operation:
  $$ (X * K)(i, j) = \sum_m \sum_n X(i+m, j+n) \cdot K(m, n) $$
- **ReLU-Aktivierung:** 
  Nach der Faltung werden negative Werte auf Null gesetzt:
  $$ f(x) = \max(0, x) $$
- **Softmax-Ausgabe:**
  Um die Endwerte (Logits) $z$ in Wahrscheinlichkeiten umzuwandeln, wird die Softmax-Funktion angewendet:
  $$ \hat{y}_i = \\frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}} $$
- **Kategorische Kreuzentropie (Loss-Funktion):**
  $$ \mathcal{L}(y, \hat{y}) = - \sum_{i=1}^{10} y_i \log(\hat{y}_i) $$

**3. Analyse von Input und Output:**
- **Input:** Der Input bestand aus $60.000$ Trainingsbildern und $10.000$ Testbildern (MNIST-Datensatz) der Größe $28 \\times 28$ Pixel mit $1$ Farbkanal. Die Pixelwerte wurden auf das Intervall $[0, 1]$ skaliert.
- **Output:** Nach 5 Epochen Training erreichte das Modell eine sehr hohe Genauigkeit (über 98%). Der berechnete *Test-Verlust* sank auf einen minimalen Wert. Das erste Diagramm visualisiert diesen stetig sinkenden Verlust $\mathcal{L}$ und die steigende Genauigkeit pro Epoche.
- **Beispielvorhersagen:** Das zweite Diagramm zeigt 5 Input-Bilder. Der Output ist die vom CNN berechnete Klasse (z. B. "7", "2"). Da "Pred" und "Wahr" übereinstimmen (grüner Text), hat das Modell die Merkmale erfolgreich mathematisch abstrahiert und zugeordnet.""",
    
    2: """### Mathematische Analyse und Auswertung (Exercise 2)

**1. Mathematische Problemstellung:**
Die Aufgabe besteht darin, eine diskrete zweidimensionale Faltung auf eine Eingabematrix (Bild) anzuwenden, um spezifische Merkmale wie Kanten oder Kontraste ohne maschinelles Lernen manuell zu extrahieren.

**2. Implementierte mathematische Operationen:**
- **Diskrete 2D-Faltung:** 
  Gegeben sei ein Bild $I \in \mathbb{R}^{28 \\times 28}$ und ein Filter (Kernel) $K \in \mathbb{R}^{3 \\times 3}$. Die resultierende Merkmalskarte $G$ wird durch das Verschieben des Filters über das Bild berechnet:
  $$ G(x, y) = \sum_{m=0}^{2} \sum_{n=0}^{2} I(x - m, y - n) \cdot K(m, n) $$
- **Verwendete Kernel (Matrizen):**
  - *Kantenerkennung (Sobel-ähnlich):*
    $$ K_{\\text{kante}} = \\begin{bmatrix} -1 & -1 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 1 & 1 \\end{bmatrix} $$
  - *Weichzeichner (Mittelwert):*
    $$ K_{\\text{weich}} = \\frac{1}{9} \\begin{bmatrix} 1 & 1 & 1 \\\\ 1 & 1 & 1 \\\\ 1 & 1 & 1 \\end{bmatrix} $$

**3. Analyse von Input und Output:**
- **Input:** Ein einzelnes Bild (Ziffer 5) als $28 \\times 28$ Matrix aus dem MNIST-Datensatz.
- **Output:** Drei verschiedene Ausgabematrizen (gefaltete Bilder). 
  - Der *Kantenerkennungs-Filter* verrechnet benachbarte Pixel so, dass horizontale Übergänge stark positiv (hell) oder negativ (dunkel) werden.
  - Der *Weichzeichner-Filter* glättet das Bild durch Durchschnittsbildung, wodurch die Ausgabematrix homogenere Werte annimmt.
  - Der *Schärfefilter* erhöht den Kontrast.
  Diese Ausgaben zeigen exakt den Effekt, den die erste Convolutional-Schicht in einem CNN erzielt, jedoch hier mit vordefinierten mathematischen Gewichten anstatt erlernten.""",
    
    3: """### Mathematische Analyse und Auswertung (Exercise 3)

**1. Mathematische Problemstellung:**
Wir wollen den mathematischen Zwischenzustand (die "Feature Maps") eines tiefen neuronalen Netzes visualisieren, um zu verstehen, in welchen mehrdimensionalen Vektorraum das Eingabebild nach der ersten Faltungsoperation transformiert wurde.

**2. Implementierte mathematische Operationen:**
- **Feature Map Extraktion:**
  Sei $I \in \mathbb{R}^{28 \\times 28 \\times 1}$ das Eingabebild und $W \in \mathbb{R}^{3 \\times 3 \\times 1 \\times 16}$ die gelernten Gewichtungsmatrizen der ersten Schicht (16 verschiedene Filter). Die Aktivierung $A_k$ für den $k$-ten Filter (mit Bias $b_k$) berechnet sich als:
  $$ A_k = \max(0, I * W_k + b_k) $$
  Hierbei repräsentiert $\max(0, x)$ die nicht-lineare ReLU-Aktivierungsfunktion.

**3. Analyse von Input und Output:**
- **Input:** Ein Testbild (Ziffer 7) mit Dimensionen $(1, 28, 28, 1)$.
- **Output:** Ein Tensor der Dimension $(1, 28, 28, 16)$, welcher 16 verschiedenen Ausgabebildern (Feature Maps) entspricht.
- **Auswertung der Diagramme und Statistiken:** Das Diagramm plottet diese 16 Matrizen. Man erkennt, dass jeder der 16 Filter $W_k$ auf eine andere geometrische Eigenschaft der Ziffer reagiert. Wo die Matrixwerte hoch sind (helle Bereiche im Plot), hat der Filter eine Übereinstimmung (z.B. eine schräge Kante) gefunden. Die Ausgabe der Minimal-, Maximal- und Mittelwerte im Text zeigt, dass der Wertebereich durch die ReLU-Aktivierung stets bei $0$ beginnt ($min = 0$) und je nach Filter unterschiedliche Maximalaktivierungen aufweist."""
}

fortgeschrittene_reps = {
    1: """### Mathematische Analyse und Auswertung (Exercise 1)

**1. Mathematische Problemstellung:**
Die Klassifikation von hochdimensionalen Farbbildern (CIFAR-10, $32 \\times 32 \\times 3$) erfordert ein tieferes Netzwerk. Um das Problem des "Internal Covariate Shift" (das Verschieben der Verteilung der Netzwerkgewichte) zu lösen, wenden wir mathematisch eine Batch-Normalisierung an.

**2. Implementierte mathematische Operationen:**
- **Batch Normalization:**
  Für einen Mini-Batch von Aktivierungen $X$ wird zunächst der Mittelwert $\mu_B$ und die Varianz $\sigma_B^2$ berechnet. Jeder Wert $x_i$ wird dann normalisiert:
  $$ \hat{x}_i = \\frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} $$
  Anschließend wird der Wert mit gelernten Parametern skaliert und verschoben:
  $$ y_i = \gamma \hat{x}_i + \\beta $$
- **Global Average Pooling:**
  Anstatt die Dimensionen abzuflachen (Flatten), berechnen wir den Durchschnitt über alle räumlichen Dimensionen $(H, W)$ für jeden Filterkanal $C$:
  $$ z_c = \\frac{1}{H \\times W} \sum_{i=1}^{H} \sum_{j=1}^{W} A(i, j, c) $$

**3. Analyse von Input und Output:**
- **Input:** Trainingsbilder aus CIFAR-10 der Dimension $(32, 32, 3)$, wobei die Pixel auf das Intervall $[0, 1]$ normalisiert wurden.
- **Output:** Nach 3 Epochen Training zeigen die Metriken (Loss und Accuracy) sowie der ausgegebene Klassifikationsbericht (Classification Report) die Leistung des Modells. 
- Im Output-Diagramm sehen wir, dass der Verlust fällt und die Genauigkeit für Trainings- und Testdaten steigt. Batch Normalization sorgt dafür, dass das Modell robuster lernt und die Wahrscheinlichkeit für absterbende Neuronen durch stabilisierte Eingaben verringert wird.""",
    
    2: """### Mathematische Analyse und Auswertung (Exercise 2)

**1. Mathematische Problemstellung:**
Um das mathematische Problem des Overfittings (Auswendiglernen) zu lösen, transformieren wir unseren Trainingsdatensatz iterativ. Wir vergrößern den Datensatz künstlich, indem wir stochastische (zufällige) Affine Transformationen anwenden.

**2. Implementierte mathematische Operationen:**
- **Affine Transformationen (Zufällige Rotation und Spiegelung):**
  Jedes Pixel $(x, y)$ des Eingabebildes wird durch eine Transformationsmatrix $T$ verschoben. Für eine Rotation um den zufälligen Winkel $\\theta$ gilt:
  $$ \\begin{bmatrix} x' \\\\ y' \\end{bmatrix} = \\begin{bmatrix} \cos \\theta & -\sin \\theta \\\\ \sin \\theta & \cos \\theta \\end{bmatrix} \\begin{bmatrix} x \\\\ y \\end{bmatrix} $$
- Bei einer horizontalen Spiegelung ändert sich das Vorzeichen der $x$-Koordinate entsprechend der Bildbreite.

**3. Analyse von Input und Output:**
- **Input:** Der ursprüngliche Datensatz, der aus unveränderten Bildern (z. B. Autos, Flugzeuge) besteht.
- **Output (Visualisierung):** Das erste Diagramm zeigt oben das Original und unten verschiedene augmentierte Versionen. Das CNN verarbeitet im Training nun rotierte, gezoomte und kontrast-angepasste Tensoren. 
- **Output (Vergleich der Modelle):** Das Modell ohne Augmentierung neigt dazu, sich die exakten Pixelwerte zu merken (höhere Trainingsgenauigkeit, aber schwächere Validierungsgenauigkeit). Das Diagramm vergleicht die Validierungskurven: Das Modell *mit* Augmentierung hat oft eine stabilere und letztlich leicht bessere Validierungsgenauigkeit, da es gelernt hat, die wahren mathematischen Invarianten (z. B. Form eines Autos unabhängig von der Neigung) zu erkennen.""",

    3: """### Mathematische Analyse und Auswertung (Exercise 3)

**1. Mathematische Problemstellung:**
Wir betrachten die Architektur des Modells selbst als einen Satz von Hyperparametern $\lambda$. Unser Ziel ist es, $\lambda = \{ \\text{Filteranzahl}, \\text{Kernelgröße}, \\text{Pooling-Typ} \}$ so zu optimieren, dass das Modell bei minimaler Laufzeit maximale Genauigkeit liefert.

**2. Implementierte mathematische Operationen:**
- **Max Pooling vs. Average Pooling:**
  Das Modell reduziert die Dimensionalität (z. B. mit einem $2 \\times 2$ Fenster).
  Bei *Max Pooling* wählt es den maximalen Wert:
  $$ y = \max_{i, j \in \{0, 1\}} x_{i, j} $$
  Bei *Average Pooling* berechnet es das arithmetische Mittel:
  $$ y = \\frac{1}{4} \sum_{i=0}^{1} \sum_{j=0}^{1} x_{i, j} $$

**3. Analyse von Input und Output:**
- **Input:** Ein reduziertes Set von 2000 MNIST-Bildern, die durch Modelle mit verschiedenen Kombinationen von Hyperparametern geführt werden.
- **Output (Vergleichstabelle & Streudiagramm):** Die erzeugte Tabelle und das Streudiagramm setzen die Anzahl der lernbaren Parameter (Input des Modells) und die Trainingszeit ins Verhältnis zur Testgenauigkeit (Output).
- Wir erkennen im Diagramm, dass eine Verdopplung der Filteranzahl zwar die Parameteranzahl (und damit die Zeit) massiv erhöht, aber nicht immer zu einer proportional besseren Testgenauigkeit führt. Größere Kernel ($5 \\times 5$) erhöhen die Berechnungskomplexität quadratisch im Vergleich zu $3 \\times 3$, weshalb die Laufzeit steigt. Die Analyse dieser Ausgaben hilft, das mathematische Optimum aus Performance und Zeit zu finden."""
}

experte_reps = {
    1: """### Mathematische Analyse und Auswertung (Exercise 1)

**1. Mathematische Problemstellung:**
Die Funktionsweise eines neuronalen Netzes (Black-Box) soll mathematisch transparent gemacht werden. Wir berechnen, wie sensitiv der Klassifikations-Score der korrekten Klasse $c$ auf Änderungen jedes einzelnen Pixels im Eingabebild $I$ reagiert. 

**2. Implementierte mathematische Operationen:**
- **Gradientenberechnung für Saliency Maps:**
  Sei $S_c(I)$ der unnormalisierte Ausgabescore der Klasse $c$. Die Saliency Map $M$ ist die Matrix der partiellen Ableitungen von $S_c$ bezüglich jedes Farbkanals des Bildes:
  $$ M_{i, j} = \max_{\\text{Kanal}} \left| \\frac{\partial S_c}{\partial I_{i, j, \\text{Kanal}}} \right| $$
  Dieser Gradient wird durch Backpropagation bis zum Eingabebild berechnet. Danach normalisieren wir die Matrix auf das Intervall $[0, 1]$.

**3. Analyse von Input und Output:**
- **Input:** Die untransformierten Pixel-Tensoren $I$ des Originalbildes.
- **Output:** Eine Heatmap (Saliency Map), die exakt dieselben Dimensionen wie das Eingabebild aufweist.
- In der Ausgabe-Grafik sehen wir das Original, die Saliency Map (Hot-Colormap) und das Overlay. Die leuchtenden, heißen Bereiche im Overlay zeigen die Pixel mit den höchsten Gradientenwerten. Das bedeutet mathematisch: Wenn wir diese Pixel ändern würden, würde sich der Output des Modells (die Gewissheit, dass es diese Ziffer ist) am stärksten ändern. Der Output beweist, dass das Modell gelernt hat, sich auf die gezeichneten Konturen der Ziffer zu fokussieren und den schwarzen Hintergrund als irrelevant zu ignorieren.""",

    2: """### Mathematische Analyse und Auswertung (Exercise 2)

**1. Mathematische Problemstellung:**
Die Standard-2D-Faltung ist mathematisch extrem rechenintensiv, da sie räumliche und tiefenbezogene Faltungen gleichzeitig ausführt. Die Problemstellung lautet, diese 3D-Tensormultiplikation in zwei simplere, aufeinanderfolgende Rechenschritte zu faktorisieren, um die Parameteranzahl zu minimieren.

**2. Implementierte mathematische Operationen:**
- **Depthwise Separable Convolution:**
  Anstatt mit einem Filter der Größe $M \\times M \\times C_{in} \\times C_{out}$ zu rechnen, teilen wir die Faltung auf:
  1. *Depthwise Convolution:* Faltet jeden der $C_{in}$ Eingangskanäle separat.
     $$ \\text{Parameter} = M \\times M \\times C_{in} $$
  2. *Pointwise Convolution:* Kombiniert die Kanäle mit $1 \\times 1$ Filtern zu $C_{out}$ Ausgabekanälen.
     $$ \\text{Parameter} = 1 \\times 1 \\times C_{in} \\times C_{out} $$

**3. Analyse von Input und Output:**
- **Input:** Die gleichen Bilddaten werden durch zwei verschiedene Architektur-Graphen geschickt (Standard Conv2D vs. Separable Conv2D).
- **Output:** Der Output-Tabelle und dem Balkendiagramm entnehmen wir einen direkten Vergleich der mathematischen Effizienz. Die Parameteranzahl beim Separable-CNN ist drastisch geringer (der Einsparungsfaktor ist deutlich berechnet). 
- Dennoch zeigt die Genauigkeit im Diagramm, dass das Separable-CNN eine ähnliche oder nur marginal geringere Genauigkeit erreicht. Das belegt, dass die Faktorisierung der Tensormultiplikation eine extrem effiziente mathematische Approximation darstellt, die den Output bei deutlich geringeren Rechenkosten annähernd bewahrt.""",

    3: """### Mathematische Analyse und Auswertung (Exercise 3)

**1. Mathematische Problemstellung:**
Wir modellieren den gesamten Lebenszyklus eines ML-Modells. Die erste mathematische Aufgabe besteht darin, Daten aus bekannten stochastischen Verteilungen und geometrischen Gleichungen synthetisch zu generieren, bevor wir ein tiefes Lernmodell darauf anwenden.

**2. Implementierte mathematische Operationen:**
- **Geometrische Generierung und Additives Rauschen:**
  Wir erzeugen perfekte geometrische Formen (Punkte $(x, y)$, für die bestimmte Gleichungen gelten, z.B. Kreisgleichung $(x-x_0)^2 + (y-y_0)^2 \leq r^2$).
  Um die Realität zu simulieren, fügen wir jedem Pixelwert $I(x, y)$ ein unabhängiges Gaußsches Rauschen hinzu:
  $$ \\tilde{I}(x, y) = \\text{clip}(I(x, y) + \mathcal{N}(\mu=0, \sigma^2=0.05^2), 0, 1) $$
- Anschließend trainieren wir ein CNN mit diesen verrauschten Tensoren und nutzen die Softmax-Funktion zur Klassifikation.

**3. Analyse von Input und Output:**
- **Input:** Ein künstlich generierter Datensatz (900 Bilder, $48 \\times 48 \\times 3$), der aus Vektoren mit zufälligem additiven Normalrauschen besteht.
- **Output:** Das Modell extrahiert die Invarianten (die Form) hinter dem stochastischen Rauschen. Die Testgenauigkeit im Output (oft nahe 100%) bestätigt, dass das Modell diese geometrischen Muster erfolgreich separieren kann.
- Die Ausgabegrafiken der Vorhersagen zeigen oben korrekt klassifizierte Beispiele (grün). Sollte das Rauschen die Pixelwerte mathematisch so stark verschoben haben, dass die Formgrenzen nicht mehr erkennbar sind, würde dies unten bei den "Fehlern" (rot) angezeigt werden. Die Verlustkurven beweisen die Konvergenz des Gradientenabstiegs."""
}

import os
path = r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_07_CNN\notebooks'
replace_formula_cells(os.path.join(path, 'Anfaenger.ipynb'), anfaenger_reps)
replace_formula_cells(os.path.join(path, 'Fortgeschrittene.ipynb'), fortgeschrittene_reps)
replace_formula_cells(os.path.join(path, 'Experte.ipynb'), experte_reps)
