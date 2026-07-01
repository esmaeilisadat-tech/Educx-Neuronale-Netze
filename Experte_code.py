#%matplotlib inline
# ============================================================
# educx GmbH – Neuronale Netze | Modul 2
# Lerntag 6: Keras & TensorFlow Grundlagen
# Niveau: Experten
# Aufgabe 1 von 3
# ============================================================
# Musterlösung – lauffähig in Spyder (tf_arm conda env)
# Python-Pfad: /Users/solusprime/opt/anaconda3/envs/tf_arm/bin/python
# ============================================================

import tensorflow as tf
import numpy as np
import matplotlib

import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)

# ── 1. GradientTape – Grundlagen ──────────────────────────────────────────────
print("\n── 1. Einfache Ableitung ──")
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x ** 2        # f(x) = x²
dy_dx = tape.gradient(y, x)  # f'(x) = 2x → bei x=3: 6
print(f"f(x)  = x²  bei x=3: {y.numpy()}")
print(f"f'(x) = 2x  bei x=3: {dy_dx.numpy()} (erwartet: 6.0)")

# ── 2. Zweite Ableitung (Gradient des Gradienten) ─────────────────────────────
print("\n── 2. Zweite Ableitung ──")
x = tf.Variable(3.0)
# Äußeres Tape aufzeichnen, damit der Gradient selbst differenzierbar ist
with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = x ** 3          # f(x)  = x³
    dy_dx  = tape1.gradient(y, x)   # f'(x) = 3x²
d2y_dx2 = tape2.gradient(dy_dx, x)  # f''(x) = 6x
print(f"f(x)   = x³   bei x=3: {y.numpy():.2f}")
print(f"f'(x)  = 3x²  bei x=3: {dy_dx.numpy():.2f}  (erwartet: 27.0)")
print(f"f''(x) = 6x   bei x=3: {d2y_dx2.numpy():.2f}  (erwartet: 18.0)")

# ── 3. Hessian-Matrix für f(x, y) = x² + 2xy + y³ ───────────────────────────
print("\n── 3. Hessian-Berechnung ──")

def f(v):
    """f(x, y) = x² + 2xy + y³"""
    x, y = v[0], v[1]
    return x**2 + 2*x*y + y**3

def hessian_berechnen(func, v):
    """Berechnet die Hessian-Matrix (Matrix zweiter Ableitungen)."""
    n = len(v)
    H = np.zeros((n, n))
    v_var = [tf.Variable(float(vi)) for vi in v]
    
    for i in range(n):
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                z = func(v_var)
            grad = tape1.gradient(z, v_var)
        grad_i_grad = tape2.gradient(grad[i], v_var)
        for j in range(n):
            H[i, j] = grad_i_grad[j].numpy() if grad_i_grad[j] is not None else 0.0
    return H

punkt = [1.0, 2.0]
H = hessian_berechnen(f, punkt)
print(f"Hessian von f(x,y) = x² + 2xy + y³ an Punkt (1, 2):")
print(f"  [[d²f/dx², d²f/dxdy], [d²f/dydx, d²f/dy²]]")
print(f"  H = {H}")
print(f"  (erwartet: [[2, 2], [2, 12]] – denn d²f/dy² = 6y = 12)")

# ── 4. Kleines Netzwerk und Gradienten-Visualisierung ────────────────────────
print("\n── 4. Gradienten je Layer visualisieren ──")

# Netzwerk erstellen
modell = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(20,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1,  activation="sigmoid"),
], name="Demo_Netz")

modell.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

# Zufällige Eingabe
x_demo = tf.constant(np.random.randn(1, 20).astype("float32"))
y_demo = tf.constant([[1.0]])

# Vorwärtsdurchlauf mit Gradienten-Aufzeichnung
with tf.GradientTape() as tape:
    vorhersage = modell(x_demo, training=True)
    verlust    = tf.keras.losses.binary_crossentropy(y_demo, vorhersage)

# Gradienten für alle Trainingsgewichte berechnen
gradienten = tape.gradient(verlust, modell.trainable_variables)

print("\nGradientenbeträge (L2-Norm) pro Layer:")
grad_normen = []
layer_namen = []
for grad, var in zip(gradienten, modell.trainable_variables):
    if grad is not None:
        norm = float(tf.norm(grad).numpy())
        grad_normen.append(norm)
        layer_namen.append(var.name)
        print(f"  {var.name:40s} L2-Norm: {norm:.6f}")

# ── 5. Visualisierung ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Hessian-Visualisierung
im = axes[0].imshow(H, cmap="RdBu", aspect="auto")
axes[0].set_title("Hessian-Matrix\nf(x,y) = x² + 2xy + y³ an (1, 2)")
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["x", "y"])
axes[0].set_yticklabels(["x", "y"])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f"{H[i,j]:.1f}", ha="center", va="center",
                     color="black", fontsize=14)
plt.colorbar(im, ax=axes[0])

# Gradienten-Normen
axes[1].barh(range(len(grad_normen)), grad_normen, color="steelblue")
axes[1].set_yticks(range(len(layer_namen)))
axes[1].set_yticklabels([n.split("/")[-1] for n in layer_namen], fontsize=9)
axes[1].set_title("Gradient L2-Normen\npro Layer (ein Vorwärtsdurchlauf)")
axes[1].set_xlabel("L2-Norm des Gradienten")
axes[1].grid(True, axis="x", alpha=0.3)

plt.suptitle("GradientTape: Zweite Ableitungen & Hessian", fontsize=13)
plt.tight_layout()
plt.savefig("E6_1_gradient_tape.png", dpi=100)
plt.show()
print("Diagramm gespeichert: E6_1_gradient_tape.png")

#%matplotlib inline
# ============================================================
# educx GmbH – Neuronale Netze | Modul 2
# Lerntag 6: Keras & TensorFlow Grundlagen
# Niveau: Experten
# Aufgabe 2 von 3
# ============================================================
# Musterlösung – lauffähig in Spyder (tf_arm conda env)
# Python-Pfad: /Users/solusprime/opt/anaconda3/envs/tf_arm/bin/python
# ============================================================

import tensorflow as tf
import numpy as np
import matplotlib

import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)

# ── 1. Daten vorbereiten ──────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:8000].astype("float32") / 255.0
y_train = y_train[:8000]
x_test  = x_test[:2000].astype("float32")  / 255.0
y_test  = y_test[:2000]

# Flatten
x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)
y_train_oh = tf.keras.utils.to_categorical(y_train, 10)
y_test_oh  = tf.keras.utils.to_categorical(y_test,  10)

# ── 2. Modell definieren ──────────────────────────────────────────────────────
def modell_erstellen():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu",    input_shape=(784,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10,  activation="softmax"),
    ])

modell = modell_erstellen()

# Optimizer und Verlustfunktion
optimizer   = tf.keras.optimizers.Adam(learning_rate=0.001)
verlustfunk = tf.keras.losses.CategoricalCrossentropy()

# ── 3. Metriken manuell tracken ───────────────────────────────────────────────
def berechne_metriken(y_wahr, y_pred_prob, n_klassen=10):
    """Berechnet Precision, Recall und F1 manuell (makro-Durchschnitt)."""
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_wahr, axis=1)
    precision_liste = []
    recall_liste    = []
    f1_liste        = []
    for k in range(n_klassen):
        tp = np.sum((y_pred == k) & (y_true == k))
        fp = np.sum((y_pred == k) & (y_true != k))
        fn = np.sum((y_pred != k) & (y_true == k))
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        precision_liste.append(p)
        recall_liste.append(r)
        f1_liste.append(f1)
    return (np.mean(precision_liste),
            np.mean(recall_liste),
            np.mean(f1_liste))

# ── 4. Batch-Training mit tf.data.Dataset ────────────────────────────────────
BATCH_GROESSE = 128
EPOCHEN       = 10

train_ds = (tf.data.Dataset
            .from_tensor_slices((x_train, y_train_oh))
            .shuffle(8000)
            .batch(BATCH_GROESSE)
            .prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset
          .from_tensor_slices((x_test, y_test_oh))
          .batch(BATCH_GROESSE))

# ── 5. Trainingsschritt mit @tf.function ──────────────────────────────────────
@tf.function
def trainingsschritt(x_batch, y_batch):
    """Ein Trainings-Schritt: forward + backward pass."""
    with tf.GradientTape() as tape:
        vorhersagen = modell(x_batch, training=True)
        verlust     = verlustfunk(y_batch, vorhersagen)
    gradienten = tape.gradient(verlust, modell.trainable_variables)
    optimizer.apply_gradients(zip(gradienten, modell.trainable_variables))
    return verlust, vorhersagen

@tf.function
def validierungsschritt(x_batch, y_batch):
    """Ein Validierungsschritt: nur forward pass."""
    vorhersagen = modell(x_batch, training=False)
    verlust     = verlustfunk(y_batch, vorhersagen)
    return verlust, vorhersagen

# ── 6. Haupttrainings-Loop ────────────────────────────────────────────────────
print("Starte benutzerdefinierten Trainings-Loop...")

verlauf = {"train_loss": [], "val_loss": [],
           "precision": [], "recall": [], "f1": []}

for epoche in range(EPOCHEN):
    # --- Training ---
    train_verluste = []
    for x_b, y_b in train_ds:
        v, _ = trainingsschritt(x_b, y_b)
        train_verluste.append(float(v))
    mittl_train_verlust = np.mean(train_verluste)

    # --- Validierung ---
    val_verluste  = []
    alle_pred     = []
    alle_wahr     = []
    for x_b, y_b in val_ds:
        v, preds = validierungsschritt(x_b, y_b)
        val_verluste.append(float(v))
        alle_pred.append(preds.numpy())
        alle_wahr.append(y_b.numpy())
    mittl_val_verlust = np.mean(val_verluste)
    alle_pred_np = np.vstack(alle_pred)
    alle_wahr_np = np.vstack(alle_wahr)

    # Metriken berechnen
    p, r, f1 = berechne_metriken(alle_wahr_np, alle_pred_np)

    verlauf["train_loss"].append(mittl_train_verlust)
    verlauf["val_loss"].append(mittl_val_verlust)
    verlauf["precision"].append(p)
    verlauf["recall"].append(r)
    verlauf["f1"].append(f1)

    print(f"Epoche {epoche+1:2d}/{EPOCHEN} | "
          f"Train-Loss: {mittl_train_verlust:.4f} | "
          f"Val-Loss: {mittl_val_verlust:.4f} | "
          f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")

print("\nTraining abgeschlossen!")
print(f"Finale F1-Score (makro): {verlauf['f1'][-1]:.4f}")

# ── 7. Visualisierung ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Verlust
axes[0].plot(verlauf["train_loss"], label="Training-Loss")
axes[0].plot(verlauf["val_loss"],   label="Validierungs-Loss")
axes[0].set_title("Verlust (Custom Training Loop)")
axes[0].set_xlabel("Epoche")
axes[0].set_ylabel("Verlust")
axes[0].legend()
axes[0].grid(True)

# Metriken
epochen_x = range(1, EPOCHEN + 1)
axes[1].plot(epochen_x, verlauf["precision"], label="Precision", marker="o")
axes[1].plot(epochen_x, verlauf["recall"],    label="Recall",    marker="s")
axes[1].plot(epochen_x, verlauf["f1"],        label="F1-Score",  marker="^", linewidth=2)
axes[1].set_title("Benutzerdefinierte Metriken (makro)")
axes[1].set_xlabel("Epoche")
axes[1].set_ylabel("Wert")
axes[1].legend()
axes[1].grid(True)
axes[1].set_ylim(0, 1.05)

plt.suptitle("Custom Training Loop mit @tf.function und manuellen Metriken",
             fontsize=13)
plt.tight_layout()
plt.savefig("E6_2_custom_training_loop.png", dpi=100)
plt.show()
print("Diagramm gespeichert: E6_2_custom_training_loop.png")

#%matplotlib inline
# ============================================================
# educx GmbH – Neuronale Netze | Modul 2
# Lerntag 6: Keras & TensorFlow Grundlagen
# Niveau: Experten
# Aufgabe 3 von 3
# ============================================================
# Musterlösung – lauffähig in Spyder (tf_arm conda env)
# Python-Pfad: /Users/solusprime/opt/anaconda3/envs/tf_arm/bin/python
# ============================================================

import tensorflow as tf
import numpy as np
import time
import matplotlib

import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)

# ── 1. Modell und Testdaten erstellen ─────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test_flat = x_test.reshape(-1, 784)[:1000]

modell = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation="relu", input_shape=(784,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10,  activation="softmax"),
], name="Benchmark_Modell")

modell.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

# Schnelltraining für Gewichte
x_tr = x_train[:5000].reshape(-1, 784).astype("float32") / 255.0
modell.fit(x_tr, y_train[:5000], epochs=3, batch_size=128, verbose=1)
print("Modell trainiert.")

# ── 2. Benchmark: Eager vs tf.function ───────────────────────────────────────
N_WIEDERHOLUNGEN = 100
x_batch = x_test_flat[:32]  # Batch der Größe 32

# Eager Mode
def eager_inferenz(x):
    return modell(x, training=False)

# tf.function – kompiliert in einen Graphen
@tf.function
def graph_inferenz(x):
    return modell(x, training=False)

print(f"\nBenchmark: {N_WIEDERHOLUNGEN} Vorhersagen mit Batch-Größe 32")

# Aufwärmen (JIT-Kompilierung beim ersten Aufruf)
_ = eager_inferenz(x_batch)
_ = graph_inferenz(x_batch)

# Eager-Zeitmessung
start = time.perf_counter()
for _ in range(N_WIEDERHOLUNGEN):
    eager_inferenz(x_batch)
eager_zeit = (time.perf_counter() - start) * 1000

# Graph-Zeitmessung
start = time.perf_counter()
for _ in range(N_WIEDERHOLUNGEN):
    graph_inferenz(x_batch)
graph_zeit = (time.perf_counter() - start) * 1000

beschleunigung = eager_zeit / graph_zeit

print(f"Eager-Modus:       {eager_zeit:.1f} ms gesamt | "
      f"{eager_zeit/N_WIEDERHOLUNGEN:.2f} ms/Vorhersage")
print(f"tf.function:       {graph_zeit:.1f} ms gesamt | "
      f"{graph_zeit/N_WIEDERHOLUNGEN:.2f} ms/Vorhersage")
print(f"Beschleunigung:    {beschleunigung:.2f}×")

# ── 3. Mixed Precision (float16) ─────────────────────────────────────────────
print("\n── 3. Mixed Precision (float16) ──")

# Mixed Precision Policy setzen
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    policy = tf.keras.mixed_precision.global_policy()
    print(f"Aktuelle Policy: {policy.name}")

    modell_fp16 = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation="relu", input_shape=(784,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        # Ausgabeschicht: float32 für numerische Stabilität
        tf.keras.layers.Dense(10, activation="softmax", dtype="float32"),
    ], name="Mixed_Precision_Modell")

    modell_fp16.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy"
    )

    # Zeitmessung fp16
    x_fp16 = tf.cast(x_batch, tf.float16)
    _ = modell_fp16(x_fp16)  # Aufwärmen

    start = time.perf_counter()
    for _ in range(N_WIEDERHOLUNGEN):
        modell_fp16(x_fp16)
    fp16_zeit = (time.perf_counter() - start) * 1000

    print(f"Mixed Precision:   {fp16_zeit:.1f} ms gesamt | "
          f"{fp16_zeit/N_WIEDERHOLUNGEN:.2f} ms/Vorhersage")

    # Policy zurücksetzen
    tf.keras.mixed_precision.set_global_policy("float32")
except Exception as e:
    print(f"Mixed Precision nicht verfügbar: {e}")
    fp16_zeit = graph_zeit  # Fallback

# ── 4. Speicherschätzung ──────────────────────────────────────────────────────
print("\n── 4. Speicherverbrauch-Schätzung ──")
def speicher_schaetzen(m, dtype_bytes=4):
    """Schätzt den Speicherverbrauch eines Modells in MB."""
    gesamt_params = sum(np.prod(v.shape) for v in m.trainable_variables)
    mb = (gesamt_params * dtype_bytes) / (1024 ** 2)
    return gesamt_params, mb

params_fp32, mb_fp32 = speicher_schaetzen(modell, 4)
print(f"Modell (float32): {params_fp32:,} Parameter → ~{mb_fp32:.2f} MB")
print(f"Modell (float16): {params_fp32:,} Parameter → ~{mb_fp32/2:.2f} MB")
print(f"Modell (int8):    {params_fp32:,} Parameter → ~{mb_fp32/4:.2f} MB")

# ── 5. Timing-Tabelle ausgeben ────────────────────────────────────────────────
print("\n── Timing-Tabelle ──")
print(f"{'Methode':<25} {'Gesamt (ms)':>15} {'Pro Inferenz (ms)':>20} {'Speedup':>10}")
print("-" * 72)
print(f"{'Eager Mode':<25} {eager_zeit:>15.1f} {eager_zeit/N_WIEDERHOLUNGEN:>20.3f} {'1.00×':>10}")
print(f"{'tf.function':<25} {graph_zeit:>15.1f} {graph_zeit/N_WIEDERHOLUNGEN:>20.3f} {beschleunigung:>9.2f}×")
print(f"{'Mixed Precision':<25} {fp16_zeit:>15.1f} {fp16_zeit/N_WIEDERHOLUNGEN:>20.3f} "
      f"{eager_zeit/fp16_zeit:>9.2f}×")
print(f"\nParameter: {params_fp32:,} | Speicher float32: {mb_fp32:.2f} MB")

# ── 6. Visualisierung ─────────────────────────────────────────────────────────
methoden   = ["Eager Mode", "tf.function", "Mixed Precision"]
zeiten     = [eager_zeit, graph_zeit, fp16_zeit]
farben     = ["#e74c3c", "#2ecc71", "#3498db"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Balkendiagramm Zeiten
bars = axes[0].bar(methoden, zeiten, color=farben, edgecolor="black")
axes[0].set_title(f"Inferenzzeit ({N_WIEDERHOLUNGEN} Durchläufe, Batch=32)")
axes[0].set_ylabel("Zeit (ms)")
axes[0].set_xlabel("Methode")
for bar, z in zip(bars, zeiten):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{z:.1f}", ha="center", va="bottom", fontsize=10)
axes[0].grid(True, axis="y", alpha=0.3)

# Speicher-Vergleich
labels  = ["float32", "float16", "int8"]
mem_mb  = [mb_fp32, mb_fp32/2, mb_fp32/4]
axes[1].bar(labels, mem_mb, color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="black")
axes[1].set_title("Modell-Speicherverbrauch (geschätzt)")
axes[1].set_ylabel("Speicher (MB)")
axes[1].set_xlabel("Datentyp")
for i, (l, m) in enumerate(zip(labels, mem_mb)):
    axes[1].text(i, m + 0.01, f"{m:.2f} MB", ha="center", va="bottom", fontsize=10)
axes[1].grid(True, axis="y", alpha=0.3)

plt.suptitle("TensorFlow Profiling und Optimierung", fontsize=14)
plt.tight_layout()
plt.savefig("E6_3_profiling.png", dpi=100)
plt.show()
print("Diagramm gespeichert: E6_3_profiling.png")

