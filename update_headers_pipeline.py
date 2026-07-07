import nbformat
import glob
import re

new_header = """# 🎯 Ziel dieser Lektion: Convolutional Neural Networks (CNN)

**Was wollen wir erreichen?** 
Wir wollen lernen, wie man neuronale Netze baut, die "sehen" können. Wir nutzen Faltungsnetzwerke (CNNs), um räumliche Muster in Bildern (wie Kanten, Formen und Texturen) zu erkennen, anstatt Bilder einfach nur abzuflachen.

**Von wo nach wo? (Problem & Ziel)** 
- **Ausgangssituation (Gegeben):** Ein Bild-Datensatz (z.B. MNIST-Ziffern oder CIFAR-10) als 2D- bzw. 3D-Tensoren (Pixel und Farbkanäle), bei denen einfache Netze (MLPs) räumliche Informationen verlieren würden.
- **Endziel (Gesucht):** Ein tiefes CNN-Modell, das durch Faltung und Pooling automatisch visuelle Merkmale extrahiert und komplexe Bilder mit hoher Genauigkeit klassifiziert.

### 🛤️ Ablauf (Schritt-für-Schritt)
`Bild-Tensoren laden` ➔ `Faltungsschichten (Conv2D) anwenden` ➔ `Dimensionen mit Pooling reduzieren` ➔ `Merkmale abflachen (Flatten/GlobalAverage)` ➔ `Wahrscheinlichkeiten klassifizieren (Dense/Softmax)`

---
"""

for filepath in glob.glob(r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_07_CNN\notebooks\*.ipynb'):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # find first markdown cell
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            # Remove old Ziel / Zweck text if it exists
            source = re.sub(r'\*\*Ziel / Zweck:\*\*.*?(\n\n|$)', '', cell.source, flags=re.DOTALL)
            # Remove existing header if we already added it (idempotency)
            source = re.sub(r'# 🎯 Ziel dieser Lektion:.*?\-\-\-\n+', '', source, flags=re.DOTALL)
            
            # Prepend new header
            cell.source = new_header + "\n" + source.strip()
            break # only do first cell

    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

print("Updated headers for all Day 7 notebooks.")
