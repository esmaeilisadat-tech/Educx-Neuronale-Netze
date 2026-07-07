import nbformat

def update_experte():
    experte_tables = {
        1: """### Verwendete Codes und Funktionen

| Code / Funktion | Zeile für Zeile Erklärung (Deutsch) |
|---|---|
| `import tensorflow as tf` | Importiert TensorFlow für maschinelles Lernen. |
| `import numpy as np` | Importiert NumPy für mathematische Matrixberechnungen. |
| `import matplotlib.pyplot as plt` | Importiert Matplotlib für die grafische Ausgabe. |
| `x_train[:5000]` | Verringert die Trainingsdaten auf 5000 Bilder, um Rechenzeit zu sparen. |
| `tf.keras.Sequential([...])` | Definiert ein sequentielles Modell. |
| `tf.keras.Input(shape=(28, 28, 1))` | Legt die Eingabeform auf 28x28 mit 1 Farbkanal fest. |
| `tf.keras.layers.Conv2D(...)`, `MaxPooling2D(...)`, `Flatten(...)`, `Dense(...)` | Erstellen die Architektur des Netzwerks (Faltung, Pooling, Abflachung, Klassifikation). |
| `modell.compile(...)` | Konfiguriert den Optimierer und die Verlustfunktion. |
| `modell.fit(..., epochs=3)` | Trainiert das Modell über 3 Epochen. |
| `probe_bild = x_train[0:1]` | Wählt ein einzelnes Bild aus dem Datensatz aus. |
| `wahre_klasse = y_train[0]` | Speichert das korrekte Label (Klasse) des Bildes ab. |
| `bild_tensor = tf.convert_to_tensor(probe_bild)` | Konvertiert das NumPy-Array in einen TensorFlow-Tensor, damit wir Ableitungen berechnen können. |
| `with tf.GradientTape() as tape:` | Öffnet eine Umgebung ("Tape"), die alle folgenden Operationen aufzeichnet, um danach mathematisch abzuleiten. |
| `tape.watch(bild_tensor)` | Sagt dem Tape explizit, dass es Änderungen am Inputbild überwachen soll. |
| `vorhersagen = modell(bild_tensor)` | Lässt das Bild durch das Modell laufen und berechnet die Klassenvorhersagen. |
| `loss = vorhersagen[0, wahre_klasse]` | Greift auf den Vorhersage-Score der *korrekten* Klasse zu (diesen wollen wir maximieren). |
| `gradienten = tape.gradient(loss, bild_tensor)` | Berechnet die partiellen Ableitungen des Loss-Wertes bezüglich jedes einzelnen Pixels im Bild. |
| `gradienten = tf.abs(gradienten)` | Nimmt den mathematischen Betrag (absoluten Wert) der Ableitungen, da uns nur die Stärke der Änderung interessiert, nicht das Vorzeichen. |
| `saliency_map = tf.reduce_max(gradienten, axis=-1)` | Reduziert die Farbkanäle, indem jeweils der höchste Gradientenwert genommen wird. |
| `saliency_map = saliency_map[0].numpy()` | Wandelt den berechneten Tensor zurück in ein normales NumPy-Array für die Visualisierung. |
| `saliency_map /= saliency_map.max()` | Normalisiert die Werte der Saliency Map auf das Intervall [0, 1]. |
| `axes[1].imshow(saliency_map, cmap="hot")` | Zeichnet die Matrix der Gradienten als Temperatur-Karte (Hot-Map) mit hellen Farben für hohe Gradienten. |
| `axes[2].imshow(saliency_map, cmap="hot", alpha=0.5)` | Legt die Saliency Map halbtransparent (alpha=0.5) über das Originalbild. |""",
        2: """### Verwendete Codes und Funktionen

| Code / Funktion | Zeile für Zeile Erklärung (Deutsch) |
|---|---|
| `def baue_modell(separable=False):` | Definiert eine Funktion, um entweder ein Standard-CNN oder ein CNN mit faktorisierten (separable) Faltungen zu bauen. |
| `tf.keras.layers.SeparableConv2D(...)` | Verwendet eine tiefenweise trennbare Faltungsschicht. Diese führt erst räumliche und dann kanalweise Multiplikationen getrennt aus, um massiv Parameter zu sparen. |
| `modell_std = baue_modell(separable=False)` | Generiert das Standard-Referenzmodell. |
| `modell_sep = baue_modell(separable=True)` | Generiert das optimierte Separable-Modell. |
| `time.perf_counter()` | Startet/Stoppt die Stoppuhr. |
| `params_std = modell_std.count_params()` | Liest die exakte mathematische Anzahl an gelernten Gewichten aus dem Standardmodell aus. |
| `params_sep = modell_sep.count_params()` | Liest die Gewichtsanzahl des optimierten Modells aus. |
| `(params_std - params_sep) / params_std * 100` | Berechnet prozentual, wie viel Speicherplatz/Variablen durch die Faktorisierung eingespart wurden. |
| `history_std = modell_std.fit(..., epochs=3)` | Trainiert das konventionelle Modell für 3 Epochen und speichert die Zeit. |
| `history_sep = modell_sep.fit(..., epochs=3)` | Trainiert das optimierte Modell für 3 Epochen. |
| `acc_std = modell_std.evaluate(...)` | Misst die finale Genauigkeit des ersten Modells auf den Testdaten. |
| `acc_sep = modell_sep.evaluate(...)` | Misst die Genauigkeit des zweiten Modells. |
| `axes[1].bar(x_pos - breite/2, werte_std, breite)` | Zeichnet im Diagramm die Balken für das erste Modell (Standard) leicht nach links verschoben ein. |
| `axes[1].bar(x_pos + breite/2, werte_sep, breite)` | Zeichnet die Balken für das zweite Modell (Separable) leicht nach rechts verschoben direkt daneben ein, um beide zu vergleichen. |
| `axes[1].set_xticks(x_pos)` | Setzt die Beschriftung der Kategorien ("Parameter", "Test-Acc", "Zeit") mittig unter die Balken-Paare. |""",
        3: """### Verwendete Codes und Funktionen

| Code / Funktion | Zeile für Zeile Erklärung (Deutsch) |
|---|---|
| `from PIL import Image, ImageDraw` | Importiert Funktionen, um leere Bilder zu erzeugen und geometrische Formen darauf zu zeichnen. |
| `def form_zeichnen(form, bildgroesse=48, rauschen=True):` | Hilfsfunktion, die abhängig vom Wert `form` (0=Kreis, 1=Quadrat, 2=Dreieck) ein Bild berechnet. |
| `img = Image.new("RGB", ...)` | Erstellt ein leeres Bild mit grauem Hintergrund ($200, 200, 200$). |
| `draw.ellipse(...)`, `draw.rectangle(...)`, `draw.polygon(...)` | Zeichnet Kreis, Quadrat oder Dreieck an den berechneten Mittelpunkt-Koordinaten. |
| `np.random.normal(0, 0.05, bild.shape)` | Erzeugt eine Matrix mit Gaußschem Rauschen (Mittelwert 0, Standardabweichung 0.05), um Sensorrauschen zu simulieren. |
| `np.clip(bild, 0, 1)` | Stellt sicher, dass das Rauschen die Pixelwerte nicht über 1 oder unter 0 verschiebt. |
| `for klasse in range(N_KLASSEN):` | Schleife, die 3 Mal durchläuft, um für jede der 3 Formen Bilder zu erzeugen. |
| `for _ in range(300):` | Erzeugt pro Klasse exakt 300 Bilder. |
| `X_list.append(bild)`, `y_list.append(klasse)` | Speichert die berechnete Bild-Matrix und die zugehörige Klasse (Zahl) in Listen. |
| `idx = np.random.permutation(len(X))` | Mischt die Reihenfolge der 900 Bilder zufällig, damit das CNN nicht zuerst alle Kreise lernt. |
| `X_train, X_test = X[:split], X[split:]` | Teilt den synthetischen Datensatz bei 80% für Training und 20% für den Test auf. |
| `tf.keras.layers.GlobalAveragePooling2D()` | Reduziert den Tensor flach vor der letzten Klassifikationsschicht. |
| `modell.fit(..., epochs=3)` | Trainiert das Netzwerk für 3 Epochen auf den synthetischen Rausch-Bildern. |
| `vorhersagen = np.argmax(modell.predict(X_test), axis=1)` | Bestimmt die Klasse mit der höchsten Wahrscheinlichkeit für jedes Testbild. |
| `falsch_idx = np.where(vorhersagen != y_test)[0]` | Sucht im NumPy-Array alle Indizes, bei denen die Vorhersage nicht dem echten Label entspricht (Fehler). |
| `for j, idx in enumerate(falsch_idx[:6]):` | Zeigt bis zu 6 Bilder an, die das Modell falsch klassifiziert hat. |"""
    }

    experte_formulas = {
        1: """### Mathematische Analyse und Auswertung (Exercise 1)

**1. Mathematische Problemstellung:**
Wir analysieren ein Black-Box-CNN, indem wir berechnen, wie sich die Ausgabe $\hat{y}$ (Klassenscore) ändert, wenn wir ein einzelnes Pixel im Eingabebild $I \in \mathbb{R}^{28 \\times 28 \\times 1}$ variieren. Dies zeigt uns, welche Pixel für die Entscheidung des Netzwerks mathematisch relevant sind.

**2. Implementierte mathematische Operationen:**
- **Saliency Maps (Gradienten bezüglich der Eingabe):**
  Sei $S_{wahre\_klasse}(I)$ der unnormalisierte Ausgabescore der korrekten Klasse (z.B. Ziffer 5). Die Saliency Map $M$ ist die Matrix der partiellen Ableitungen:
  $$ M_{i, j} = \\left| \\frac{\partial S_5}{\partial I_{i, j}} \\right| $$
  Wenn der Gradient am Pixel $(10, 10)$ z.B. $12.5$ beträgt, würde eine kleine Änderung an diesem Pixel den Score für die Ziffer 5 extrem stark verändern. Ein Gradient nahe $0$ bedeutet, das Pixel ist irrelevant (Hintergrund).
- Wir nehmen den Betrag $| \cdot |$, da uns sowohl stark positive als auch stark negative Änderungen interessieren, und normalisieren die Werte durch Division durch das Maximum auf das Intervall $[0, 1]$.

**3. Analyse von Input und Output:**
- **Input:** Ein untransformierter Pixel-Tensor der Größe $28 \\times 28 \\times 1$ aus den MNIST-Trainingsdaten.
- **Output:** Eine Heatmap der Größe $28 \\times 28$.
- Im generierten Diagramm zeigt die Hot-Colormap die Saliency Map. Die leuchtenden, heißen Bereiche (Werte nahe $1.0$) decken sich exakt mit den gezeichneten Konturen der Ziffer. Der schwarze Hintergrund (Werte nahe $0.0$) hat Gradienten von Null. Das beweist mathematisch, dass das CNN die Konturen zur Klassifikation nutzt und gelernt hat, Hintergrundrauschen zu ignorieren.""",
        2: """### Mathematische Analyse und Auswertung (Exercise 2)

**1. Mathematische Problemstellung:**
Die Standard-Faltung ist rechenintensiv, da sie alle Farbkanäle und alle benachbarten Pixel (räumlich) in einer einzigen 3D-Operation multipliziert. Wir faktorisieren diese Operation in zwei 2D-Schritte, um die Parameter $\\theta$ drastisch zu reduzieren.

**2. Implementierte mathematische Operationen:**
- **Standard Conv2D vs. Depthwise Separable Convolution:**
  Ein normaler $3 \\times 3$ Filter mit $32$ Eingangs- und $64$ Ausgangskanälen benötigt:
  $$ \\text{Parameter}_{std} = 3 \\cdot 3 \\cdot 32 \\cdot 64 = 18.432 $$
  Die *Depthwise Separable Convolution* teilt dies auf:
  1. *Depthwise:* Faltet jeden der $32$ Eingangskanäle separat.
     $$ 3 \\cdot 3 \\cdot 32 = 288 \\text{ Parameter} $$
  2. *Pointwise:* Kombiniert die $32$ Kanäle mit $1 \\times 1$ Filtern zu $64$ Ausgangskanälen.
     $$ 1 \\cdot 1 \\cdot 32 \\cdot 64 = 2.048 \\text{ Parameter} $$
  $$ \\text{Parameter}_{sep} = 288 + 2.048 = 2.336 $$
  Das ist eine theoretische Einsparung von fast $\approx 87\%$.

**3. Analyse von Input und Output:**
- **Input:** Jeweils $5000$ CIFAR-10 Bilder der Größe $32 \\times 32 \\times 3$.
- **Output:** Die Konsolentabelle und die Balkendiagramme setzen die exakten mathematischen Kennzahlen ins Verhältnis.
- Der Output vergleicht die tatsächliche Modellgröße. Das Standardmodell hat z.B. $\approx 112.000$ Parameter, das Separable-Modell nur $\approx 14.000$. Obwohl das Modell $\approx 87\%$ kleiner ist, erreichen beide Modelle im rechten Diagramm eine fast identische Genauigkeit ($\approx 55-60\%$). Das zeigt, dass die mathematische Annäherung durch Faktorisierung extrem effizient ist.""",
        3: """### Mathematische Analyse und Auswertung (Exercise 3)

**1. Mathematische Problemstellung:**
Bevor Modelle auf reale Daten angewandt werden, müssen wir oft kontrollierte Umgebungen schaffen. Wir generieren Bildtensoren $I$ nach strengen geometrischen Regeln und addieren künstliches Rauschen (Störsignale), um das Modell auf Robustheit zu testen.

**2. Implementierte mathematische Operationen:**
- **Geometrische Generierung und Rauschen:**
  Für einen Kreis berechnen wir alle Punkte $(x, y)$, die die euklidische Distanzgleichung zum Mittelpunkt $(m, m)$ erfüllen:
  $$ (x-m)^2 + (y-m)^2 \leq r^2 $$
  Diese Pixel erhalten den RGB-Wert $(50, 100, 200)$, skaliert auf $[0, 1]$.
  Um Sensorrauschen zu simulieren, addieren wir für JEDES Pixel $(x, y)$ einen Wert aus der Normalverteilung $\mathcal{N}$:
  $$ I_{neu}(x, y) = I(x, y) + z, \quad z \sim \mathcal{N}(0, 0.05^2) $$
  Werte über $1.0$ oder unter $0.0$ werden rigoros abgeschnitten (Clipping).

**3. Analyse von Input und Output:**
- **Input:** Ein Datensatz aus $900$ Bild-Tensoren der Größe $48 \\times 48 \\times 3$, der in $720$ Trainings- und $180$ Testdaten aufgeteilt wird.
- **Output:** Nach 3 Epochen meldet das Netz eine Genauigkeit nahe $100\%$ ($\approx 1.0000$). Das Netz hat gelernt, die mathematischen Invarianten (die Form) hinter dem Gaußschen Rauschen zu erkennen.
- Die Visualisierungen zeigen die Beispiele. In der "Vorhersagen"-Matrix (grüner Text) stimmen die Ausgaben fast perfekt mit der Wahrheit überein. Falls ein generiertes Bild extrem stark verrauscht war und das Modell sich irrte, taucht es in der unteren Zeile in Rot auf. Der schnelle Konvergenzverlauf im Verlust-Graphen bestätigt, dass die Convolutional Layers prädestiniert dafür sind, diese räumlichen Muster zu isolieren."""
    }

    import os
    filepath = r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_07_CNN\notebooks\Experte.ipynb'
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    t_idx = 1
    f_idx = 1
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            if '### Verwendete Codes und Funktionen' in cell.source:
                if t_idx in experte_tables:
                    cell.source = experte_tables[t_idx]
                t_idx += 1
            if '### Mathematische Analyse und Auswertung' in cell.source:
                if f_idx in experte_formulas:
                    cell.source = experte_formulas[f_idx]
                f_idx += 1

    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

update_experte()
