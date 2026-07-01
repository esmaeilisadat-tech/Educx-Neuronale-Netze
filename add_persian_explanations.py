import json
import os

NOTEBOOKS = [
    'c:/Users/esmae/Documents/Educx Neuronale Netze/NN_Projekt_Workspace/01_Neural_Network_Basics/notebooks/Anfaenger.ipynb',
    'c:/Users/esmae/Documents/Educx Neuronale Netze/NN_Projekt_Workspace/01_Neural_Network_Basics/notebooks/Fortgeschrittene.ipynb',
    'c:/Users/esmae/Documents/Educx Neuronale Netze/NN_Projekt_Workspace/01_Neural_Network_Basics/notebooks/Experte.ipynb'
]

TEXTS = {
    1: [
        "### Mathematische Erklärungen und Code-Ablauf (Übung 1: Biologisches Neuron und gewichtete Summe)\n",
        "\n",
        "In dieser Zelle haben wir die grundlegende Funktion eines Neurons simuliert. Der Prozess sieht so aus, dass jede Eingabe ($x_i$) mit einem bestimmten Gewicht ($w_i$) multipliziert und dann mit einem Bias-Wert ($b$) addiert wird. Diese Operation zeigt den Einfluss jeder Eingabe auf die endgültige Ausgabe des Neurons.\n",
        "\n",
        "Die grundlegende mathematische Formel für diese Berechnungen sieht wie folgt aus:\n",
        "\n",
        r"$$ z = \sum_{i=1}^{n} (x_i \cdot w_i) + b $$" + "\n",
        "\n",
        "- **$x_i$**: Eingabewerte (Signale, die von Dendriten empfangen werden)\n",
        r"- **$w_i$**: Zugehörige Gewichte für jede Eingabe (Stärke der Synapsen)" + "\n",
        "- **$b$**: Bias (Startpunkt oder Aktivierungsschwelle)\n",
        "- **$z$**: Lineare Ausgabe des Neurons\n",
        "\n",
        "**Diagramm:**\n",
        "Das in diesem Abschnitt gezeichnete Diagramm zeigt normalerweise die Eingabewerte, Gewichte und das Ergebnis ihrer linearen Kombination. Dieses Bild hilft uns zu verstehen, wie verschiedene Eingabewerte basierend auf ihren Gewichten die endgültige Ausgabe formen.\n"
    ],
    2: [
        "### Mathematische Erklärungen und Code-Ablauf (Übung 2: Aktivierungsfunktion)\n",
        "\n",
        "In dieser Zelle haben wir der linearen Ausgabe des Neurons eine kontinuierliche Aktivierungsfunktion (wie die Sigmoid-Funktion) hinzugefügt. Aktivierungsfunktionen sind notwendig, um Nichtlinearität in neuronalen Netzen zu erzeugen, damit das Netzwerk komplexere Muster lernen kann.\n",
        "\n",
        "Die mathematische Formel der Sigmoid-Funktion lautet wie folgt:\n",
        "\n",
        r"$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$" + "\n",
        "\n",
        "- **$z$**: Die lineare Ausgabe des Neurons aus dem vorherigen Schritt.\n",
        "- **$e$**: Eulersche Zahl (ungefähr 2,718).\n",
        r"- **$\sigma(z)$**: Die endgültige Ausgabe des Neurons, die immer eine Zahl zwischen 0 und 1 sein wird." + "\n",
        "\n",
        "**Diagramm:**\n",
        "Das in diesem Abschnitt gezeichnete Diagramm ist eine S-förmige Kurve (Sigmoid-Kurve). Die horizontale Achse (X) zeigt die Eingabewerte der Funktion ($z$) und die vertikale Achse (Y) die Ausgabewerte (zwischen 0 und 1). Dieses Diagramm zeigt sehr gut, wie sehr große oder sehr kleine Eingabewerte in einen bestimmten Bereich komprimiert werden, um die Wahrscheinlichkeit oder Intensität der Aktivierung des Neurons zu bestimmen.\n"
    ],
    3: [
        "### Mathematische Erklärungen und Code-Ablauf (Übung 3: Berechnung des Fehlers oder Loss)\n",
        "\n",
        "In dieser Zelle befassen wir uns mit dem Konzept der Fehlerberechnung (Loss). Damit ein neuronales Netz lernen kann, muss es wissen, wie weit seine Vorhersagen von der Realität entfernt sind. Die Fehlerfunktion misst diesen Abstand. Eine der häufigsten Fehlerfunktionen ist die mittlere quadratische Abweichung (Mean Squared Error - MSE).\n",
        "\n",
        "Die mathematische Formel für MSE lautet wie folgt:\n",
        "\n",
        r"$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$" + "\n",
        "\n",
        "- **$n$**: Gesamtzahl der Datenpunkte\n",
        "- **$y_i$**: Tatsächlicher Wert (Target)\n",
        r"- **$\hat{y}_i$**: Vom neuronalen Netz vorhergesagter Wert" + "\n",
        r"- **$(y_i - \hat{y}_i)$**: Vorhersagefehler für jeden Datenpunkt (Residual)" + "\n",
        "\n",
        "**Diagramm:**\n",
        r"Das hier gezeichnete Diagramm zeigt die Differenz zwischen den tatsächlichen Werten und den Vorhersagen des Modells (z. B. als Balkendiagramm, um die Fehlermenge in jedem Beispiel zu zeigen). Mathematisch gesehen visualisiert dieses Diagramm den Abstand zwischen $y$ und $\hat{y}$. Das ultimative Ziel beim Training des Netzwerks ist es, dieses Fehlerniveau im Diagramm zu minimieren." + "\n"
    ]
}

def clean_and_process_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # First, remove existing injected markdown cells from previous runs
    cleaned_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            source = "".join(cell.get('source', []))
            if 'توضیحات' in source or 'Mathematische Erklärungen' in source:
                continue # Skip the injected cell
        cleaned_cells.append(cell)

    new_cells = []
    exercise_num = 0
    added_for_exercise = set()
    
    for cell in cleaned_cells:
        new_cells.append(cell)
        
        # Track exercise number from markdown cells
        if cell['cell_type'] == 'markdown':
            source = "".join(cell.get('source', []))
            if 'Exercise 1' in source:
                exercise_num = 1
            elif 'Exercise 2' in source:
                exercise_num = 2
            elif 'Exercise 3' in source:
                exercise_num = 3
                
        # If it's a code cell, add explanation if not already added for this exercise
        elif cell['cell_type'] == 'code':
            if exercise_num in TEXTS and exercise_num not in added_for_exercise:
                persian_cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": TEXTS[exercise_num]
                }
                new_cells.append(persian_cell)
                added_for_exercise.add(exercise_num)
                
    nb['cells'] = new_cells
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"Processed {os.path.basename(path)} - Added explanations for exercises: {list(added_for_exercise)}")

if __name__ == '__main__':
    for p in NOTEBOOKS:
        if os.path.exists(p):
            clean_and_process_notebook(p)
        else:
            print(f"File not found: {p}")
