import nbformat; import json; 
def process_notebook(filepath, content_data):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    new_cells = []
    ex_idx = 1
    
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and cell.source.startswith('## Exercise'):
            ex_data = content_data.get(ex_idx)
            if ex_data:
                ziel_text = f'\n\n**Ziel / Zweck:**\n{ex_data["ziel"]}\n'
                cell.source += ziel_text
            new_cells.append(cell)
        elif cell.cell_type == 'code':
            ex_data = content_data.get(ex_idx)
            if ex_data:
                table_md = nbformat.v4.new_markdown_cell(source=ex_data['table'])
                new_cells.append(table_md)
                
                source = cell.source
                source = source.replace('epochs=20', 'epochs=3')
                source = source.replace('epochs=15', 'epochs=3')
                source = source.replace('epochs=10', 'epochs=3')
                source = source.replace('epochs=8', 'epochs=3')
                source = source.replace('EPOCHEN = 8', 'EPOCHEN = 3')
                cell.source = source
                
                new_cells.append(cell)
                
                formula_md = nbformat.v4.new_markdown_cell(source=ex_data['formula'])
                new_cells.append(formula_md)
                
                ex_idx += 1
            else:
                new_cells.append(cell)
        else:
            new_cells.append(cell)
            
    nb.cells = new_cells
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

anfaenger_data = {
    1: {
        'ziel': 'In dieser Aufgabe lernen wir, wie man ein grundlegendes Convolutional Neural Network (CNN) mit Keras und TensorFlow von Grund auf neu aufbaut. Der Zweck ist es, zu verstehen, wie Bilder (hier der MNIST-Datensatz mit handgeschriebenen Ziffern) verarbeitet und durch Faltungs- und Pooling-Schichten geleitet werden, um Merkmale zu extrahieren und letztendlich die Ziffern erfolgreich zu klassifizieren.',
        'table': '### Verwendete Codes und Funktionen\n\n| Code / Funktion | Beschreibung (Deutsch) |\n|---|---|\n| `tf.keras.datasets.mnist.load_data()` | Lädt den MNIST-Datensatz mit Trainings- und Testdaten. |\n| `np.newaxis` | Fügt eine neue Dimension hinzu (z. B. für den Farbkanal). |\n| `tf.keras.Sequential([...])` | Erstellt ein sequentielles Modell (Schichten werden nacheinander ausgeführt). |\n| `tf.keras.layers.Conv2D(...)` | Führt eine zweidimensionale Faltung auf dem Eingabebild durch. |\n| `tf.keras.layers.MaxPooling2D(...)` | Reduziert die räumlichen Dimensionen durch Auswahl des Maximalwerts. |\n| `tf.keras.layers.Flatten()` | Wandelt die mehrdimensionalen Merkmalskarten in einen Vektor um. |\n| `tf.keras.layers.Dense(...)` | Erstellt eine vollständig verbundene (Dense) Schicht. |\n| `modell.compile(...)` | Konfiguriert das Modell für das Training (Optimierer, Verlustfunktion). |\n| `modell.fit(...)` | Trainiert das Modell auf den Trainingsdaten. |\n| `modell.evaluate(...)` | Bewertet das Modell auf den Testdaten. |\n| `modell.predict(...)` | Erzeugt Vorhersagen für gegebene Eingabedaten. |\n| `np.argmax(...)` | Gibt den Index des Maximalwerts zurück (vorhergesagte Klasse). |',
        'formula': '### Mathematische Formel\n\n**1. 2D-Faltungsoperation (Convolution):**\nDas elementweise Produkt zwischen dem Bildausschnitt $I$ und dem Filter (Kernel) $K$ summiert über alle Kanäle:\n\n$$ (I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) K(m, n) $$\n\n**2. Softmax-Aktivierungsfunktion (für Klassifikation):**\nWandelt die letzten Netzwerkausgaben (Logits) $z$ in Wahrscheinlichkeiten um:\n\n$$ \sigma(\\mathbf{z})_i = \\frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}} $$'
    },
    2: {
        'ziel': 'Das Ziel dieser Aufgabe ist es, die Funktionsweise von Faltungsfiltern (Kernels) visuell greifbar zu machen. Wir wenden manuell verschiedene Filtermatrizen (z. B. für Kantenerkennung, Weichzeichnung und Schärfung) auf ein einziges Bild an. So sehen wir genau, was im ersten Schritt eines Convolutional Neural Networks passiert, bevor das Netzwerk die optimalen Filterwerte selbständig durch Training erlernt.',
        'table': '### Verwendete Codes und Funktionen\n\n| Code / Funktion | Beschreibung (Deutsch) |\n|---|---|\n| `bild = x_train[0]` | Wählt das erste Bild aus dem Trainingsdatensatz aus. |\n| `np.array(...)` | Erstellt ein NumPy-Array zur Definition der Filtermatrizen (Kernel). |\n| `convolve(bild, f, ...)` | Wendet die Filtermatrix `f` auf das Bild `bild` an (2D-Faltungsoperation). |\n| `gefaltetes_bild.min() / .max()` | Ermittelt den kleinsten bzw. größten Pixelwert im gefalteten Bild. |\n| `plt.subplots(...)` | Erstellt ein Raster von Subplots zur Darstellung mehrerer Bilder. |\n| `ax.imshow(...)` | Zeigt das Bild oder die Matrix grafisch in einem Subplot an. |\n| `zip(axes[1:], ergebnisse)` | Verknüpft die restlichen Achsen mit den Ergebnissen der Faltungen. |',
        'formula': '### Mathematische Formel\n\n**Diskrete 2D-Faltung (ohne Padding und Stride=1):**\nEin Filter $K$ der Größe $M \\times N$ wird auf ein Bild $I$ angewendet, um das Ausgangsbild $G$ zu erhalten:\n\n$$ G(x, y) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} I(x - m, y - n) K(m, n) $$'
    },
    3: {
        'ziel': 'In dieser Aufgabe möchten wir in das Innere unseres trainierten CNNs blicken. Der Zweck ist es, die sogenannten „Feature Maps“ (Merkmalskarten) zu extrahieren und zu visualisieren, die das Netzwerk während der Analyse erzeugt. Dadurch wird verständlicher, auf welche Muster und Kanten das neuronale Netz achtet, wenn es Entscheidungen trifft.',
        'table': '### Verwendete Codes und Funktionen\n\n| Code / Funktion | Beschreibung (Deutsch) |\n|---|---|\n| `tf.keras.Model(inputs=..., outputs=...)` | Erstellt ein Keras-Modell, das bestimmte Zwischenschichten ausgibt. |\n| `modell.get_layer("conv_1").output` | Greift auf die Ausgabe der spezifischen Schicht namens `conv_1` zu. |\n| `feature_map_modell.predict(...)` | Liefert in diesem Fall die extrahierten Feature Maps für ein Bild. |\n| `feature_maps.shape[-1]` | Liest die Anzahl der Kanäle (Filter) aus der Form des Tensors ab. |\n| `math.ceil(...)` | Rundet eine Zahl auf die nächste ganze Zahl auf. |\n| `axes[zeile, spalte].imshow(...)` | Zeigt eine einzelne Feature Map im entsprechenden Rasterfeld an. |\n| `fm.mean()` | Berechnet den Durchschnittswert der Feature Map. |',
        'formula': '### Mathematische Formel\n\n**Feature Map Aktivierung:**\nDie Aktivierungskarte (Feature Map) $A_k$ für Filter $K_k$ nach Anwendung der Faltung und Hinzufügen des Bias $b_k$, gefolgt von der ReLU-Aktivierungsfunktion $f(x) = \max(0, x)$ wird berechnet als:\n\n$$ A_k = f(I * K_k + b_k) $$'
    }
}

fortgeschrittene_data = {
    1: {
        'ziel': 'Das Ziel dieser Aufgabe ist der Aufbau eines tieferen CNNs zur Bildklassifikation des CIFAR-10 Datensatzes (Farbbilder in 10 Kategorien). Wir lernen erweiterte Techniken wie Batch Normalization und Global Average Pooling kennen. Der Zweck dieser Architektur ist es, eine höhere Genauigkeit zu erreichen und das Netzwerk robuster sowie effizienter trainierbar zu machen.',
        'table': '### Verwendete Codes und Funktionen\n\n| Code / Funktion | Beschreibung (Deutsch) |\n|---|---|\n| `tf.keras.datasets.cifar10.load_data()` | Lädt den CIFAR-10 Datensatz (Farbbilder in 10 Klassen). |\n| `tf.keras.layers.BatchNormalization()` | Normalisiert die Ausgaben der Schicht für ein stabileres Training. |\n| `tf.keras.layers.Activation("relu")` | Wendet die ReLU-Aktivierungsfunktion separat nach der Normalisierung an. |\n| `tf.keras.layers.Dropout(...)` | Setzt zufällig Neuronen auf 0, um Overfitting zu vermeiden. |\n| `tf.keras.layers.GlobalAveragePooling2D()` | Bildet den Durchschnitt über räumliche Dimensionen (ersetzt Flatten). |\n| `tf.keras.callbacks.ReduceLROnPlateau(...)` | Reduziert die Lernrate, wenn die Genauigkeit stagniert. |\n| `classification_report(...)` | Erstellt einen Textbericht über die Klassifikationsleistung. |',
        'formula': '### Mathematische Formel\n\n**Batch Normalization:**\nFür einen Mini-Batch wird jeder Wert $x^{(k)}$ bezüglich des Mittelwerts $\\mu_B$ und der Varianz $\\sigma_B^2$ normalisiert, um das Training zu beschleunigen:\n\n$$ \hat{x}^{(k)} = \\frac{x^{(k)} - \\mu_B}{\sqrt{\\sigma_B^2 + \\epsilon}} $$\n$$ y^{(k)} = \\gamma^{(k)} \hat{x}^{(k)} + \\beta^{(k)} $$'
    },
    2: {
        'ziel': 'In dieser Übung befassen wir uns mit \'Data Augmentation\' (Datenaugmentierung). Der Zweck liegt darin, durch zufällige Transformationen (Spiegeln, Drehen, Zoomen) der vorhandenen Trainingsbilder den Datensatz künstlich zu vergrößern. Dadurch soll das Netzwerk verallgemeinern lernen und das Overfitting (Auswendiglernen der Daten) verringern.',
        'table': '### Verwendete Codes und Funktionen\n\n| Code / Funktion | Beschreibung (Deutsch) |\n|---|---|\n| `tf.keras.layers.RandomFlip(...)` | Spiegelt Bilder zufällig horizontal zur Datenerweiterung. |\n| `tf.keras.layers.RandomRotation(...)` | Rotiert Bilder zufällig um einen bestimmten Winkel. |\n| `tf.keras.layers.RandomZoom(...)` | Zoomt zufällig in Bilder hinein oder heraus. |\n| `tf.keras.layers.RandomContrast(...)` | Ändert zufällig den Kontrast der Bilder. |\n| `tf.clip_by_value(aug_bild, 0, 1)` | Begrenzt die Pixelwerte auf den gültigen Bereich zwischen 0 und 1. |',
        'formula': '### Mathematische Formel\n\n**Geometrische Transformation (Rotation):**\nEine Drehung eines Pixels an der Position $(x, y)$ um den Winkel $\\theta$ wird mathematisch über folgende Rotationsmatrix beschrieben:\n\n$$ \\begin{bmatrix} x\' \\\\ y\' \end{bmatrix} = \\begin{bmatrix} \cos \\theta & -\sin \\theta \\\\ \sin \\theta & \cos \\theta \end{bmatrix} \\begin{bmatrix} x \\\\ y \end{bmatrix} $$'
    },
    3: {
        'ziel': 'Diese Aufgabe demonstriert, wie sich verschiedene Hyperparameter (wie die Anzahl der Filter, Kernelgrößen und Pooling-Methoden) auf die Trainingszeit und Genauigkeit eines CNNs auswirken. Der Zweck ist es, durch gezieltes Ausprobieren (eine kleine Form des Hyperparameter-Tunings) die effizienteste und beste Architektur für das jeweilige Problem zu finden.',
        'table': '### Verwendete Codes und Funktionen\n\n| Code / Funktion | Beschreibung (Deutsch) |\n|---|---|\n| `tf.keras.layers.AveragePooling2D(...)` | Reduziert die Bildgröße durch Berechnung des Durchschnitts im Fenster. |\n| `time.perf_counter()` | Startet oder stoppt eine genaue Zeitmessung (für Trainingsdauer). |\n| `m.count_params()` | Gibt die Gesamtanzahl der Parameter des Modells zurück. |\n| `axes[0].scatter(...)` | Erstellt ein Streudiagramm (Scatterplot) zum Parameter-Vergleich. |\n| `axes[0].annotate(...)` | Fügt Textbeschriftungen (Namen der Modelle) in den Scatterplot ein. |',
        'formula': '### Mathematische Formel\n\n**Average Pooling:**\nFür ein $2 \\times 2$ Fenster aggregiert Average Pooling die Werte im Gegensatz zu MaxPooling durch Bildung des arithmetischen Mittels:\n\n$$ y_{i, j} = \\frac{1}{4} \sum_{m=0}^{1} \sum_{n=0}^{1} x_{2i+m, 2j+n} $$'
    }
}

experte_data = {
    1: {
        'ziel': 'Hier lernen wir, sogenannte \'Saliency Maps\' zu erstellen. Ziel ist es, Entscheidungen des CNNs interpretierbar zu machen (Explainable AI). Wir berechnen, wie sensibel die Netzwerkvorhersage auf Änderungen einzelner Bildpixel reagiert. So finden wir heraus, auf welche Bereiche im Bild das Netzwerk besonders geachtet hat, um seine Klassifikation zu treffen.',
        'table': '### Verwendete Codes und Funktionen\n\n| Code / Funktion | Beschreibung (Deutsch) |\n|---|---|\n| `tf.Variable(..., dtype=tf.float32)` | Wandelt das Bild in eine Variable um, sodass Gradienten berechnet werden können. |\n| `tf.GradientTape()` | Zeichnet Operationen auf, um die automatische Differenzierung durchzuführen. |\n| `tape.gradient(score, bild_tensor)` | Berechnet den Gradienten des Klassenscores in Bezug auf die Eingangspixel. |\n| `tf.reduce_max(...)` | Bestimmt den maximalen Gradientenbetrag über alle Farbkanäle hinweg. |\n| `tf.abs(...)` | Berechnet den Absolutbetrag der Gradienten. |',
        'formula': '### Mathematische Formel\n\n**Saliency Map (Gradientenberechnung):**\nSei $S_c(I)$ der Klassenscore der vorhergesagten Klasse $c$ für das Bild $I$. Die Saliency Map $M$ ergibt sich aus dem Betrag der partiellen Ableitungen des Scores bezüglich der Bildpixel:\n\n$$ M_{i, j} = \max_{c\'} \\left| \\frac{\\partial S_c}{\\partial I_{i, j, c\'}} \\right| $$'
    },
    2: {
        'ziel': 'In dieser Expertenaufgabe vergleichen wir eine Standard-2D-Faltung mit einer recheneffizienteren \'Depthwise Separable Convolution\'. Das Ziel ist es, die Architektur zu optimieren, indem wir die Anzahl der trainierbaren Parameter und die Rechenzeit drastisch reduzieren, ohne dabei zu viel an Genauigkeit einzubüßen. Solche Techniken werden vor allem für Modelle auf mobilen Geräten (MobileNet) eingesetzt.',
        'table': '### Verwendete Codes und Funktionen\n\n| Code / Funktion | Beschreibung (Deutsch) |\n|---|---|\n| `tf.keras.layers.SeparableConv2D(...)` | Führt eine tiefenweise separierbare Faltung durch (deutlich weniger Parameter). |\n| `axes[1].bar(...)` | Erstellt ein Balkendiagramm, um Modelle direkt zu vergleichen. |\n| `np.arange(...)` | Erzeugt ein Array mit gleichmäßig verteilten Werten für die Balken. |',
        'formula': '### Mathematische Formel\n\n**Depthwise Separable Convolution:**\nAnstatt alle Kanäle und Pixel gleichzeitig zu falten, wird die Operation in zwei Schritte unterteilt. Für $C$ Eingangskanäle, $F$ Filter und Filtergröße $M \\times M$:\n\n1. Depthwise Convolution (pro Kanal einzeln):\n   Parameter $= M \\times M \\times C$\n2. Pointwise Convolution (Kombination über $1 \\times 1$ Filter):\n   Parameter $= 1 \\times 1 \\times C \\times F$\n\n$$ P_{\\text{sep}} = M^2 C + CF \\quad \\text{vs.} \\quad P_{\\text{std}} = M^2 C F $$'
    },
    3: {
        'ziel': 'Zum Abschluss bauen wir eine End-to-End-Lösung mit einem selbst generierten synthetischen Datensatz. Das Ziel ist es, zu zeigen, wie man Bilder (Kreise, Quadrate, Dreiecke) per Code erstellt, mit Rauschen versieht und anschließend ein CNN darauf trainiert. Der Zweck besteht darin, den gesamten Workflow der Datenbeschaffung, Modellierung und Visualisierung auf neue Problemstellungen anwenden zu können.',
        'table': '### Verwendete Codes und Funktionen\n\n| Code / Funktion | Beschreibung (Deutsch) |\n|---|---|\n| `Image.new("RGB", ...)` | Erstellt ein neues RGB-Bild mit PIL (Python Imaging Library). |\n| `ImageDraw.Draw(img)` | Initialisiert ein Objekt zum Zeichnen geometrischer Formen. |\n| `draw.ellipse(...)` | Zeichnet eine Ellipse (oder einen Kreis) auf das Bild. |\n| `draw.rectangle(...)` | Zeichnet ein Rechteck (oder ein Quadrat). |\n| `draw.polygon(...)` | Zeichnet ein beliebiges Polygon basierend auf einer Liste von Eckpunkten. |\n| `np.random.normal(...)` | Fügt Gaußsches Rauschen hinzu, um Varianz im Datensatz zu erzeugen. |\n| `np.clip(...)` | Begrenzt die Array-Werte, damit sie trotz Rauschen im Intervall [0, 1] bleiben. |\n| `np.random.permutation(...)` | Erzeugt eine zufällig gemischte Reihenfolge von Indizes. |',
        'formula': '### Mathematische Formel\n\n**Additives Gaußsches Rauschen:**\nUm realistische Bilddaten zu simulieren, wird Rauschen, das einer Normalverteilung folgt, zum Pixelwert des Bildes $I(x, y)$ addiert:\n\n$$ \\tilde{I}(x, y) = I(x, y) + \mathcal{N}(\\mu, \\sigma^2) $$\n\nwobei in diesem Code $\\mu = 0$ und $\\sigma = 0.05$ verwendet werden.'
    }
}

import os
path = r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_07_CNN\notebooks'
process_notebook(os.path.join(path, 'Anfaenger.ipynb'), anfaenger_data)
process_notebook(os.path.join(path, 'Fortgeschrittene.ipynb'), fortgeschrittene_data)
process_notebook(os.path.join(path, 'Experte.ipynb'), experte_data)
