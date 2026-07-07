import nbformat

def update_notebooks():
    anfaenger_tables = {
        1: """### Verwendete Codes und Funktionen

| Code / Funktion | Zeile für Zeile Erklärung (Deutsch) |
|---|---|
| `import tensorflow as tf` | Importiert die TensorFlow-Bibliothek für maschinelles Lernen unter dem Kürzel `tf`. |
| `import numpy as np` | Importiert NumPy für numerische Operationen und Arrays unter dem Kürzel `np`. |
| `import matplotlib.pyplot as plt` | Importiert das Modul zum Zeichnen von Graphen und Diagrammen unter dem Kürzel `plt`. |
| `tf.__version__` | Gibt die aktuell installierte Version von TensorFlow aus. |
| `tf.keras.datasets.mnist.load_data()` | Lädt den MNIST-Datensatz (Trainings- und Testdaten mit 60.000 bzw. 10.000 Bildern) herunter. |
| `astype("float32")` | Konvertiert die Pixel-Integerwerte (0-255) in Fließkommazahlen für präzisere Berechnungen. |
| `[..., np.newaxis]` | Fügt den Bildern eine weitere Dimension hinzu (Farbkanal-Dimension, aus 28x28 wird 28x28x1). |
| `/ 255.0` | Normalisiert die Pixelwerte mathematisch von [0, 255] auf [0, 1]. |
| `tf.keras.Sequential([...])` | Erstellt ein Feed-Forward-Netzwerk (ein Modell, in dem die Daten Schicht für Schicht weitergegeben werden). |
| `tf.keras.Input(shape=(28, 28, 1))` | Definiert die exakte Form des Eingabebildes (28 Pixel Höhe, 28 Pixel Breite, 1 Farbkanal). |
| `tf.keras.layers.Conv2D(32, (3, 3), activation="relu")` | Fügt eine Faltungsschicht mit 32 Filtern der Größe 3x3 hinzu. Die ReLU-Aktivierungsfunktion setzt negative Werte auf 0. |
| `tf.keras.layers.MaxPooling2D((2, 2))` | Verkleinert die Feature Maps um den Faktor 2 (nimmt den maximalen Wert aus 2x2 Feldern). |
| `tf.keras.layers.Flatten()` | Flacht den mehrdimensionalen Tensor in einen 1D-Vektor ab, um ihn an die Dense-Schicht weiterzugeben. |
| `tf.keras.layers.Dense(128, activation="relu")` | Fügt eine vollständig verknüpfte Schicht mit 128 Neuronen und ReLU-Aktivierung hinzu. |
| `tf.keras.layers.Dense(10, activation="softmax")` | Erstellt die Ausgabeschicht mit 10 Neuronen (für 10 Ziffern). Softmax wandelt die Werte in Wahrscheinlichkeiten um. |
| `modell.compile(...)` | Konfiguriert den Lernprozess: "adam" als Optimierer, Crossentropy als Verlustfunktion, Accuracy als Metrik. |
| `modell.summary()` | Gibt eine Übersichtstabelle des Modells mit allen Schichten und der Parameteranzahl aus. |
| `modell.fit(..., epochs=5, batch_size=128, validation_split=0.1)` | Startet das Training für 5 Epochen. 10% der Trainingsdaten werden als Validierungsdaten abgespalten. Die Batchgröße ist 128. |
| `modell.evaluate(x_test, y_test)` | Testet das finale Modell auf den Testdaten und berechnet den Verlust sowie die Genauigkeit. |
| `plt.subplots(1, 2, figsize=(12, 4))` | Erstellt eine Zeichenfläche mit 1 Zeile und 2 nebeneinander liegenden Diagrammen (Größe 12x4). |
| `axes[0].plot(history.history["loss"])` | Zeichnet die Verlust-Werte (Loss) des Trainings über die Epochen in das erste Diagramm. |
| `plt.savefig("...", dpi=100)` | Speichert die gezeichneten Diagramme als PNG-Bilddatei mit einer Auflösung von 100 dpi. |
| `modell.predict(x_test[:5])` | Lässt das Modell Vorhersagen (Wahrscheinlichkeiten) für die ersten 5 Testbilder treffen. |
| `np.argmax(vorhersagen, axis=1)` | Sucht in den Wahrscheinlichkeiten nach dem höchsten Wert (die wahrscheinlichste Klasse). |
| `axes[i].imshow(x_test[i, :, :, 0], cmap="gray")` | Zeichnet das Bild im Graustufenmodus auf den Bildschirm. |""",
        2: """### Verwendete Codes und Funktionen

| Code / Funktion | Zeile für Zeile Erklärung (Deutsch) |
|---|---|
| `from scipy.ndimage import convolve` | Importiert die `convolve`-Funktion, um eine manuelle 2D-Faltung (Matrixmultiplikation) durchzuführen. |
| `x_train[0].astype("float32")` | Wählt das erste Bild aus den Trainingsdaten aus und konvertiert es für die Berechnung in Dezimalzahlen. |
| `np.array([...], dtype="float32")` | Erstellt einen 3x3 NumPy-Array (Matrix), der als unser manueller Filter/Kernel fungiert. |
| `filter_liste = [...]` | Erstellt eine Liste von Tupeln, die den Filternamen und die dazugehörige Matrix für eine Iteration speichert. |
| `ergebnisse = []` | Initialisiert eine leere Liste, um die gefalteten Ausgabebilder später zu speichern. |
| `for name, f in filter_liste:` | Startet eine Schleife, die jeden Namen und jeden Filter (Matrix) aus der Liste durchgeht. |
| `convolve(bild, f, mode="constant", cval=0.0)` | Wendet den 3x3 Filter `f` auf das `bild` an. Ränder werden mit 0en aufgefüllt (Padding). |
| `ergebnisse.append((name, gefaltetes_bild))` | Speichert den Namen des Filters und das berechnete Bild-Ergebnis in die Liste. |
| `gefaltetes_bild.min()` / `gefaltetes_bild.max()` | Berechnet den kleinsten und größten Pixelwert des gefalteten Bildes, um die Änderung zu sehen. |
| `fig, axes = plt.subplots(1, 4, ...)` | Erstellt ein Raster von 4 Diagrammen nebeneinander (1 Originalbild + 3 Filterergebnisse). |
| `zip(axes[1:], ergebnisse)` | Verknüpft die letzten 3 leeren Diagramm-Boxen mit den 3 zuvor berechneten Filter-Ergebnissen für die Ausgabe. |
| `ax.imshow(ergebnis, cmap="gray")` | Zeigt das gefaltete Matrix-Ergebnis als Graustufenbild an. |
| `ax.set_title(...)` / `ax.axis("off")` | Setzt den Titel über das Bild und versteckt die störenden X- und Y-Achsen-Striche. |""",
        3: """### Verwendete Codes und Funktionen

| Code / Funktion | Zeile für Zeile Erklärung (Deutsch) |
|---|---|
| `import math` | Importiert mathematische Standardfunktionen wie Aufrunden (ceil). |
| `x_train[:5000]` | Beschneidet die Trainingsdaten auf 5000 Bilder, um das Training für dieses Beispiel zu beschleunigen. |
| `tf.keras.Model(inputs=..., outputs=...)` | Erstellt ein neues, funktionales Keras-Modell, das bestimmte Zwischenschichten abgreift. |
| `modell.input` | Greift auf den Eingangstensor des zuvor erstellten `modell` zu. |
| `modell.get_layer("conv_1").output` | Sucht die Schicht mit dem Namen `conv_1` und nutzt deren Ausgabe als Output für das neue Modell. |
| `x_test[0:1]` | Extrahiert das exakt erste Testbild (die Slicing-Syntax `0:1` erhält die Form des Tensors: 1x28x28x1). |
| `feature_map_modell.predict(probe_bild)` | Lässt das Eingabebild durch das Extraktor-Modell laufen und gibt die Aktivierungen der Convolution-Schicht zurück. |
| `feature_maps.shape[-1]` | Liest das letzte Element aus der Tensor-Form ab (gibt die Anzahl der Filter / Kanäle zurück, hier 16). |
| `math.ceil(n_filter / n_spalten)` | Berechnet die Anzahl der benötigten Plot-Zeilen. Teilt die Filter durch die Spaltenzahl und rundet auf. |
| `for col in range(n_spalten):` | Iteriert über die Spalten der ersten Zeile im Plot, um das Originalbild darzustellen. |
| `for i in range(n_filter):` | Iteriert von 0 bis 15, um jede der 16 Feature Maps zu verarbeiten. |
| `zeile = i // n_spalten + 1` | Berechnet mathematisch die korrekte Zeile im Plot für den Index `i` (mittels Ganzzahldivision). |
| `spalte = i % n_spalten` | Berechnet die korrekte Spalte im Plot für den Index `i` (mittels Modulo-Operation). |
| `fm = feature_maps[0, :, :, i]` | Extrahiert aus dem 4D-Tensor exakt die `i`-te Feature Map des ersten Bildes als 2D-Matrix. |
| `axes[zeile, spalte].imshow(fm, cmap="viridis")` | Zeichnet die Feature Map mit einem Farbverlauf ("viridis", zeigt hohe und niedrige Werte deutlich an). |
| `for i in range(n_filter, n_zeilen * n_spalten):` | Iteriert über restliche, leere Diagramm-Plätze und schaltet deren Achsen ab (`axis("off")`). |
| `fm.min()`, `fm.max()`, `fm.mean()` | Berechnen das Minimum, das Maximum und den Durchschnitt der Aktivierungswerte der jeweiligen Matrix. |"""
    }

    anfaenger_formulas = {
        1: """### Mathematische Analyse und Auswertung (Exercise 1)

**1. Mathematische Problemstellung:**
Wir betrachten ein Klassifikationsproblem, bei dem ein Eingabebild $X \in \mathbb{R}^{28 \\times 28 \\times 1}$ auf einen Wahrscheinlichkeitsvektor $\hat{y} \in \mathbb{R}^{10}$ abgebildet werden soll. Das Ziel ist es, die Parameter $\\theta$ des CNNs zu optimieren.

**2. Implementierte mathematische Operationen:**
- **2D-Faltung (Convolution):** 
  Wir haben Bilder der Größe $28 \\times 28$ und Filter $K$ der Größe $3 \\times 3$. Für den ersten Pixel $(0,0)$ lautet die Operation mit echten Zahlen:
  $$ (X * K)(0, 0) = \sum_{m=0}^{2} \sum_{n=0}^{2} X(0+m, 0+n) \cdot K(m, n) $$
- **ReLU-Aktivierung:** 
  Nach der Faltung werden negative Werte auf $0$ gesetzt. Wenn der Faltungswert z.B. $-2.5$ ist, wird er zu $0$:
  $$ f(-2.5) = \max(0, -2.5) = 0 $$
- **Softmax-Ausgabe:**
  Für 10 Klassen berechnet Softmax die Wahrscheinlichkeit. Wenn die Logits $z = [1.2, 0.1, 5.0, ...]$ sind, berechnet sich die Wahrscheinlichkeit für Klasse 3 (Ziffer 2) als:
  $$ \hat{y}_3 = \\frac{e^{5.0}}{e^{1.2} + e^{0.1} + e^{5.0} + ...} \approx 0.96 $$

**3. Analyse von Input und Output:**
- **Input:** $60.000$ Bilder aus dem MNIST-Datensatz mit $28 \\times 28$ Pixeln. Die Pixelwerte wurden durch $/ 255.0$ exakt von z.B. $128$ auf $0.50$ skaliert.
- **Output:** Nach 5 Epochen Training auf diesen Werten erreichte das Modell z.B. eine Genauigkeit von $\approx 98.5\%$. Der *Test-Verlust* lag bei $\approx 0.05$. Das Diagramm visualisiert, dass Vorhersage und Wahrheit z.B. bei der Ziffer 7 identisch sind.""",
        2: """### Mathematische Analyse und Auswertung (Exercise 2)

**1. Mathematische Problemstellung:**
Eine diskrete zweidimensionale Faltung wird manuell (ohne neuronales Netz) auf eine Eingabematrix ($28 \\times 28$ Bild der Ziffer 5) angewendet, um Kanten oder Kontraste zu extrahieren.

**2. Implementierte mathematische Operationen:**
- **Diskrete 2D-Faltung:** 
  Für das Bild $I$ (Größe $28 \\times 28$) und den $3 \\times 3$ Kernel $K$ lautet die Gleichung konkret:
  $$ G(x, y) = \sum_{m=0}^{2} \sum_{n=0}^{2} I(x - m, y - n) \cdot K(m, n) $$
- **Verwendete Kernel (Matrizen) in echten Zahlen:**
  - *Kantenerkennung (Sobel horizontal):* Obere Reihe multipliziert mit $-1$, untere mit $+1$.
    $$ K_{\\text{kante}} = \\begin{bmatrix} -1 & -1 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 1 & 1 \\end{bmatrix} $$
  - *Weichzeichner (Mittelwert):* Durchschnitt aller 9 Felder.
    $$ K_{\\text{weich}} = \\begin{bmatrix} 0.111 & 0.111 & 0.111 \\\\ 0.111 & 0.111 & 0.111 \\\\ 0.111 & 0.111 & 0.111 \\end{bmatrix} $$
  - *Schärfefilter:* Zentraler Pixel wird mit $5$ multipliziert, die angrenzenden abgezogen.
    $$ K_{\\text{schaerfe}} = \\begin{bmatrix} 0 & -1 & 0 \\\\ -1 & 5 & -1 \\\\ 0 & -1 & 0 \\end{bmatrix} $$

**3. Analyse von Input und Output:**
- **Input:** Ein Matrix-Ausschnitt des $28 \\times 28$ MNIST-Bildes (Ziffer 5) mit Werten in $[0.0, 1.0]$.
- **Output:** Drei Ausgabematrizen derselben Größe. 
  - Die Multiplikation mit der *Kantenerkennung* resultiert in hohen Werten (z.B. $>1.0$) an oberen Kanten und niedrigen (z.B. $<-1.0$) an unteren Kanten. 
  - Der *Weichzeichner* mittelt die Pixel, sodass harte Kanten verschwinden (maximaler Wert sinkt, minimaler steigt).""",
        3: """### Mathematische Analyse und Auswertung (Exercise 3)

**1. Mathematische Problemstellung:**
Wir visualisieren die $16$ "Feature Maps" (Aktivierungsmatrizen) der ersten Schicht, um zu verstehen, wie das $28 \\times 28 \\times 1$ Bild in den $28 \\times 28 \\times 16$ Tensorraum projiziert wird.

**2. Implementierte mathematische Operationen:**
- **Feature Map Extraktion:**
  Das Eingabebild $I$ ($28 \\times 28$) wird mit 16 trainierten Filtern $W_1$ bis $W_{16}$ (jeweils $3 \\times 3$) gefaltet und durch die ReLU-Funktion geleitet. Für die Matrix $A_{16}$ (Filter 16) gilt:
  $$ A_{16} = \max(0, I * W_{16} + b_{16}) $$
  Jeder Pixel, bei dem das Ergebnis $I * W_{16} + b_{16}$ z.B. $-0.73$ ergab, wurde auf exakt $0.00$ gesetzt.

**3. Analyse von Input und Output:**
- **Input:** Das erste Testbild aus $x\_test[0:1]$ der Größe $(1, 28, 28, 1)$, was der Ziffer 7 entspricht.
- **Output:** Ein 4D-Tensor $feature\_maps$ mit der exakten Form $(1, 28, 28, 16)$. 
- **Auswertung der Diagramme und Statistiken:** Das Skript plottet 16 Arrays der Größe $28 \\times 28$. Da die ReLU-Aktivierung aktiv war, zeigen die Ausgaben Minimalwerte von exakt $0.000$ (`min=0.000`). Filter 1 könnte z.B. einen Maximalwert von $4.512$ aufweisen, was bedeutet, dass dort das gesuchte geometrische Muster besonders stark mit dem $3 \\times 3$ Kernel korrelierte. Helle, gelbe Bereiche repräsentieren diese hohen positiven Zahlen, dunkle Bereiche repräsentieren die $0.000$ Werte."""
    }

    import os
    filepath = r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_07_CNN\notebooks\Anfaenger.ipynb'
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    t_idx = 1
    f_idx = 1
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            if '### Verwendete Codes und Funktionen' in cell.source:
                if t_idx in anfaenger_tables:
                    cell.source = anfaenger_tables[t_idx]
                t_idx += 1
            if '### Mathematische Analyse und Auswertung' in cell.source:
                if f_idx in anfaenger_formulas:
                    cell.source = anfaenger_formulas[f_idx]
                f_idx += 1

    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

update_notebooks()
