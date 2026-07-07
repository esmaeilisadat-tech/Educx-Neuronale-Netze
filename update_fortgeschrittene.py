import nbformat

def update_fortgeschrittene():
    fortgeschrittene_tables = {
        1: """### Verwendete Codes und Funktionen

| Code / Funktion | Zeile für Zeile Erklärung (Deutsch) |
|---|---|
| `import tensorflow as tf` | Importiert die TensorFlow-Bibliothek für neuronale Netze. |
| `import numpy as np` | Importiert NumPy für numerische Matrizenoperationen. |
| `import matplotlib.pyplot as plt` | Importiert Matplotlib zur Darstellung von Graphen. |
| `from sklearn.metrics import classification_report` | Importiert eine Funktion zur genauen Auswertung der Klassifikationsergebnisse. |
| `tf.keras.datasets.cifar10.load_data()` | Lädt den CIFAR-10 Datensatz (60.000 kleine Farbbilder, aufgeteilt in 50k Training und 10k Test). |
| `x_train.astype("float32") / 255.0` | Konvertiert die Pixel (0-255) zu Fließkommazahlen und normalisiert sie in den Bereich [0, 1]. |
| `klassen_namen = [...]` | Erstellt eine Liste mit den Text-Namen der 10 Kategorien in CIFAR-10. |
| `tf.keras.Sequential([...])` | Baut ein fortlaufendes Modell auf. |
| `tf.keras.Input(shape=(32, 32, 3))` | Definiert das Eingabeformat: 32x32 Pixel, 3 Farbkanäle (RGB). |
| `tf.keras.layers.Conv2D(32, (3, 3), padding="same")` | Faltungsschicht mit 32 Filtern der Größe 3x3. `padding="same"` bedeutet, dass die Bildgröße nicht schrumpft. |
| `tf.keras.layers.BatchNormalization()` | Normalisiert die Werte (Feature Maps) des aktuellen Batches, um das Training zu beschleunigen und robuster zu machen. |
| `tf.keras.layers.Activation("relu")` | Wendet die ReLU-Aktivierungsfunktion ($f(x)=\max(0,x)$) auf die normalisierten Werte an. |
| `tf.keras.layers.MaxPooling2D((2, 2))` | Halbiert die Dimensionen der Bild-Tensoren, indem jeweils der höchste Wert in 2x2 Fenstern genommen wird. |
| `tf.keras.layers.Dropout(0.25)` | Deaktiviert zufällig 25% der Neuronen in jedem Trainingsschritt, um Overfitting zu verhindern. |
| `tf.keras.layers.GlobalAveragePooling2D()` | Bildet für jeden Farbkanal den räumlichen Durchschnitt. Reduziert den Tensor flach und macht `Flatten()` überflüssig. |
| `tf.keras.layers.Dense(10, activation="softmax")` | Ausgabe-Schicht für 10 Klassen mit Softmax für Wahrscheinlichkeitswerte. |
| `modell.compile(...)` | Konfiguriert den Optimizer (adam), Loss (sparse_categorical_crossentropy) und Metrik. |
| `modell.fit(..., epochs=3)` | Trainiert das Modell für 3 Epochen über den gesamten Trainingsdatensatz. |
| `modell.evaluate(x_test, y_test)` | Berechnet den Verlust und die Test-Genauigkeit. |
| `np.argmax(modell.predict(x_test), axis=1)` | Ermittelt die Klassen-ID mit der jeweils höchsten vorhergesagten Wahrscheinlichkeit für die Testbilder. |
| `classification_report(y_test, preds, target_names=...)` | Erstellt einen Bericht mit Precision, Recall und f1-score für jede der 10 Klassen. |
| `plt.subplots(...)`, `plot(...)`, `savefig(...)` | Funktionen zum Zeichnen, Speichern und Anzeigen der Verlust- und Genauigkeitsgraphen. |""",
        2: """### Verwendete Codes und Funktionen

| Code / Funktion | Zeile für Zeile Erklärung (Deutsch) |
|---|---|
| `tf.keras.Sequential([...], name="Augmentierung")` | Erstellt einen Pipeline-Graphen für Bild-Transformationen, die zur Laufzeit im Training passieren. |
| `tf.keras.layers.RandomRotation(0.1)` | Rotiert die Bilder zufällig um bis zu ±10% von $360^\circ$ (also $\\pm 36^\circ$). |
| `tf.keras.layers.RandomZoom(0.1)` | Zoomt das Bild zufällig hinein oder heraus (um bis zu 10%). |
| `tf.keras.layers.RandomContrast(0.1)` | Ändert den Kontrast der Bilder zufällig um $\\pm 10\%$. |
| `tf.keras.layers.RandomFlip("horizontal")` | Spiegelt das Bild mit 50% Wahrscheinlichkeit horizontal (rechts-links). |
| `def baue_modell():` | Definiert eine Hilfsfunktion, um dasselbe Basis-CNN mehrfach identisch aufzubauen, um Vergleiche zu ermöglichen. |
| `modell_ohne = baue_modell()` | Baut das Netzwerk ohne Augmentierung. |
| `modell_mit = tf.keras.Sequential([augmentierung, baue_modell()])` | Baut das Netzwerk, wobei die Bilder VOR dem Eintreffen in das CNN live augmentiert werden. |
| `modell.fit(..., validation_split=0.15)` | Startet das Training für 3 Epochen. Wir zweigen 15% der Daten zur Validierung ab, um echtes Overfitting zu messen. |
| `history_ohne.history["val_accuracy"][-1]` | Holt sich exakt den letzten Genauigkeits-Wert nach Abschluss der Epochen. |
| `(acc_mit - acc_ohne)*100` | Berechnet die Differenz der Genauigkeiten in Prozentpunkten. |
| `augmentierung(probe, training=True)` | Wendet den Augmentierungsgraphen manuell auf ein Bild an (nur im Trainingsmodus `True` wird wirklich augmentiert). |
| `tf.clip_by_value(aug_bild, 0, 1)` | Stellt sicher, dass die Pixelwerte nach dem Kontrastwechsel nicht unter 0 oder über 1 fallen (sonst Warnungen beim Plotten). |
| `axes[...].imshow(...)` | Zeichnet die erzeugten Bilder in das Subplot-Raster. |""",
        3: """### Verwendete Codes und Funktionen

| Code / Funktion | Zeile für Zeile Erklärung (Deutsch) |
|---|---|
| `import time` | Importiert das Zeit-Modul, um die Dauer des Trainings zu messen. |
| `x_train[:2000]` / `x_test[:500]` | Schneidet die Datensätze extrem ab, damit wir in annehmbarer Zeit viele CNNs durchtesten können. |
| `def cnn_variante(filter_anzahl, kernel_groesse, pooling_typ, name):` | Eine Funktion (Fabrik), die ein Modell entsprechend der übergebenen Hyperparameter (Konfiguration) generiert. |
| `if pooling_typ == "Max": ... else: ...` | Überprüft den Parameter `pooling_typ` als String und wählt die entsprechende Keras-Schichtklasse (Max vs. Average) aus. |
| `tf.keras.layers.Conv2D(...)` | Die Parameteranzahl für Filter und Kernelgröße werden nun dynamisch in die Definition des CNNs eingesetzt. |
| `kombinationen = [...]` | Eine Liste von Python-Dictionaries. Jedes Dictionary repräsentiert einen anderen Hyperparameter-Satz, der getestet werden soll. |
| `for i, k in enumerate(kombinationen):` | Schleife über alle Dictionaries (Hyperparameter-Kombinationen). `k` enthält die aktuellen Parameter. |
| `time.perf_counter()` | Stoppuhr-Funktion. Wir merken uns die Zeit vor und nach `fit()`, um die genaue Dauer des Trainings in Sekunden zu berechnen. |
| `m.count_params()` | Zählt und liefert die exakte Anzahl der mathematischen Gewichte/Parameter im aktuellen Modell. |
| `ergebnisse.append({...})` | Speichert alle Kennzahlen (Genauigkeit, Trainingszeit, Parameteranzahl) eines Laufs in eine Ergebnis-Liste. |
| `sorted(ergebnisse, key=lambda x: x["Test-Acc"], reverse=True)` | Sortiert die Ergebnisliste absteigend (von bester zu schlechtester Genauigkeit). |
| `max(ergebnisse, key=lambda x: x["Test-Acc"])` | Findet das Dictionary (den Lauf), welches die höchste Genauigkeit (`Test-Acc`) besitzt. |
| `axes[0].scatter(zeiten, akk, s=100, c=..., cmap="tab10")` | Zeichnet ein Streudiagramm (Punktwolke), bei dem auf der X-Achse die Trainingszeit und auf der Y-Achse die Genauigkeit liegt. |
| `axes[0].annotate(name, (zeiten[i], akk[i]))` | Schreibt den Konfigurations-Namen neben jeden Punkt in das Streudiagramm. |"""
    }

    fortgeschrittene_formulas = {
        1: """### Mathematische Analyse und Auswertung (Exercise 1)

**1. Mathematische Problemstellung:**
Die Klassifikation von Farbbildern aus CIFAR-10 ($32 \\times 32 \\times 3$) erfordert eine Stabilisierung der Trainingsgewichte während tiefen Faltungen, da kleine Variationen in frühen Schichten in späteren Schichten exponentiell explodieren ("Internal Covariate Shift").

**2. Implementierte mathematische Operationen:**
- **Batch Normalization:**
  Wir betrachten einen Mini-Batch von $32$ Aktivierungen. Für den Pixel-Wert $x_i$ eines Kanals wird der Mittelwert $\mu_B$ und die Varianz $\sigma_B^2$ der $32$ Bilder berechnet. Der Wert wird dann normalisiert:
  $$ \hat{x}_i = \\frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + 0.001}} $$
  (Hierbei ist $\epsilon = 0.001$). Anschließend wird der Wert mit den gelernten Schicht-Parametern skaliert und verschoben:
  $$ y_i = \gamma \cdot \hat{x}_i + \\beta $$
- **Global Average Pooling:**
  Anstatt alle räumlichen Werte zu behalten, berechnen wir für jeden Kanal exakt EINEN Durchschnittswert aus den $8 \\times 8 = 64$ verbleibenden Pixeln $A$. Für Kanal $c$ bedeutet das:
  $$ z_c = \\frac{1}{64} \sum_{i=1}^{8} \sum_{j=1}^{8} A(i, j, c) $$

**3. Analyse von Input und Output:**
- **Input:** $50.000$ Farbbilder der Dimension $(32, 32, 3)$, deren Pixelwerte von $255$ durch Skalierung auf $[0.0, 1.0]$ reduziert wurden.
- **Output:** Nach 3 Epochen meldet der `classification_report` Test-Genauigkeiten von z.B. $\approx 65-70\%$. Die ausgegebene Tabelle listet die Klassen "Frosch" (frog) oder "Auto" (automobile). Ein Wert wie `Precision=0.72` bedeutet, dass von 100 Bildern, die das Netz als Frosch klassifizierte, 72 tatsächlich Frösche waren. Die Batch Normalization sorgte für einen stabilisierten Output-Verlustkurvenverlauf (im Diagramm), weil extreme Pixelaktivierungen kontinuierlich wieder um Null zentriert wurden.""",
        2: """### Mathematische Analyse und Auswertung (Exercise 2)

**1. Mathematische Problemstellung:**
Wir erweitern die Trainingsdaten ($N=50.000$) künstlich, um Overfitting zu verhindern, indem wir zufällige mathematische Matrixtransformationen auf die Eingangsbilder anwenden.

**2. Implementierte mathematische Operationen:**
- **Affine Transformation (Zufällige Rotation `RandomRotation(0.1)`):**
  Jedes Pixel an der Position $(x, y)$ des $32 \\times 32$ Bildes wird um einen zufälligen Winkel $\\theta \in [-36^\circ, 36^\circ]$ gedreht:
  $$ \\begin{bmatrix} x' \\\\ y' \\end{bmatrix} = \\begin{bmatrix} \cos \\theta & -\sin \\theta \\\\ \sin \\theta & \cos \\theta \\end{bmatrix} \\begin{bmatrix} x \\\\ y \\end{bmatrix} $$
- **Farbtransformation (Zufälliger Kontrast `RandomContrast(0.1)`):**
  Wir passen die Kontrastwerte $C$ eines Bildes um den Faktor $\alpha \in [0.9, 1.1]$ an den Mittelwert $\bar{x}$ an:
  $$ I_{neu} = (I_{alt} - \bar{x}) \cdot \alpha + \bar{x} $$

**3. Analyse von Input und Output:**
- **Input:** $50.000$ unveränderte Bilder aus CIFAR-10.
- **Output (Bilder):** Das obere Raster zeigt ein Originalbild (z. B. Frosch). Das untere Raster zeigt den Output der Transformationen. Sie sehen 5 Variationen desselben Bildes: Leicht gedreht, gespiegelt oder verblasst.
- **Output (Leistung):** Das Netz trainiert mit echten Variablen-Veränderungen. Das erste Modell erreichte ohne Augmentierung auf den Validierungsdaten z.B. $\approx 66.5\%$. Das Modell mit Augmentierung erreicht oft ähnliche, aber über viele Epochen stabilere Genauigkeiten. Die Validierungs-Genauigkeits-Kurve "Mit Augmentierung" springt weniger, da das Netz nicht die statischen Pixelmuster des Trainingssatzes auswendig gelernt hat (Overfitting), sondern gelernt hat, die wahre Funktion (wie sieht ein Frosch aus) anzunähern.""",
        3: """### Mathematische Analyse und Auswertung (Exercise 3)

**1. Mathematische Problemstellung:**
Es gilt, den diskreten Suchraum der Hyperparameter $\lambda = \{ \\text{Filterzahl}, \\text{Kernel}, \\text{Pooling} \}$ zu durchlaufen und empirisch die Funktion $Genauigkeit = f(\lambda)$ bei gegebenem Zeitbudget auf einer Teilmenge ($N=2000$) zu optimieren.

**2. Implementierte mathematische Operationen:**
- **Max Pooling vs. Average Pooling:**
  Beim `MaxPooling2D((2,2))` wird aus jedem $2 \\times 2$ Block ($4$ Pixel) exakt einer ausgewählt:
  $$ y_{max} = \max(x_{0,0}, x_{0,1}, x_{1,0}, x_{1,1}) $$
  Beim `AveragePooling2D((2,2))` wird der Summenwert durch $4$ geteilt:
  $$ y_{avg} = \\frac{1}{4} (x_{0,0} + x_{0,1} + x_{1,0} + x_{1,1}) $$

**3. Analyse von Input und Output:**
- **Input:** Ein Set von $2000$ Bildern und $7$ verschiedene Konfigurationen (Dictionaries).
- **Output (Modell-Parameter & Zeit):** Die Tabelle druckt die exakten Berechnungen aus. Wenn wir den Kernel von $3 \\times 3$ (Fläche 9) auf $5 \\times 5$ (Fläche 25) bei $32$ Filtern vergrößern, steigt die Anzahl der mathematischen Parameter massiv an. Entsprechend meldet der Code einen Anstieg der Laufzeit (Output in Sekunden).
- **Output (Streudiagramm):** Auf der Y-Achse sehen wir die Testgenauigkeit, auf der X-Achse die benötigte Zeit. Die beste Konfiguration ist der Punkt ganz oben links (hohe Genauigkeit, niedrige Zeit). Man sieht oft, dass `F32_K3_PMax` schneller läuft als `F32_K5_PMax`, aber fast den gleichen Informationsgehalt (Test-Acc) lernt. Der Graph zeigt, dass die Optimierungsfunktion bei Convolutional Networks nicht linear mit mehr Parametern skaliert."""
    }

    import os
    filepath = r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_07_CNN\notebooks\Fortgeschrittene.ipynb'
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    t_idx = 1
    f_idx = 1
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            if '### Verwendete Codes und Funktionen' in cell.source:
                if t_idx in fortgeschrittene_tables:
                    cell.source = fortgeschrittene_tables[t_idx]
                t_idx += 1
            if '### Mathematische Analyse und Auswertung' in cell.source:
                if f_idx in fortgeschrittene_formulas:
                    cell.source = fortgeschrittene_formulas[f_idx]
                f_idx += 1

    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

update_fortgeschrittene()
