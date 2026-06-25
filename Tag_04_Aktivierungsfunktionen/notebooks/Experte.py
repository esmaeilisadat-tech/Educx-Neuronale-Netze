#!/usr/bin/env python
# coding: utf-8

# ## Exercise 1
# 
# **Dataset Used:** Custom/Synthetic Array Data (numpy)
# 
# The following code implements the steps for this exercise. Outputs and charts are generated automatically inline.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# ============================================================
# educx GmbH – Neuronale Netze | Modul 2
# Lerntag 4: Aktivierungsfunktionen
# Niveau: Experten
# Aufgabe 1 von 3
# ============================================================
# Musterlösung – lauffähig in Spyder (tf_arm conda env)
# Python-Pfad: /Users/solusprime/opt/anaconda3/envs/tf_arm/bin/python
# ============================================================

# Implement custom activation functions in TensorFlow
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import tensorflow as tf

tf.random.set_seed(42); np.random.seed(42)

# Benutzerdefinierte Aktivierungen als Keras-Layer
class Mish(tf.keras.layers.Layer):
    """Mish: x * tanh(softplus(x)) — oft besser als ReLU"""
    def call(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))

class GELU(tf.keras.layers.Layer):
    """Gaussian Error Linear Unit — verwendet in BERT, GPT"""
    def call(self, x):
        return 0.5 * x * (1.0 + tf.math.tanh(
            tf.cast(tf.sqrt(2.0/np.pi), tf.float32) * (x + 0.044715 * tf.pow(x, 3))))

class PReLU_custom(tf.keras.layers.Layer):
    """Parametric ReLU — lernbarer negativer Slope"""
    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha', shape=(1,), initializer='zeros', trainable=True)
    def call(self, x):
        return tf.where(x > 0, x, self.alpha * x)

# Numerische Implementierungen für Visualisierung
x_vals = np.linspace(-4, 4, 300).astype(np.float32)

gelu_np  = 0.5 * x_vals * (1 + np.tanh(np.sqrt(2/np.pi) * (x_vals + 0.044715 * x_vals**3)))
mish_np  = x_vals * np.tanh(np.log(1 + np.exp(np.clip(x_vals, -500, 500))))
swish_np = x_vals * (1 / (1 + np.exp(-np.clip(x_vals, -500, 500))))
relu_np  = np.maximum(0, x_vals)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_vals, relu_np,  'b-', linewidth=2, label='ReLU')
ax.plot(x_vals, mish_np,  'r-', linewidth=2, label='Mish')
ax.plot(x_vals, gelu_np,  'g-', linewidth=2, label='GELU')
ax.plot(x_vals, swish_np, 'm--', linewidth=2, label='Swish')
ax.set_title('Moderne Aktivierungsfunktionen: Mish, GELU, Swish')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.legend(); ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5); ax.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('moderne_aktivierungen.png', dpi=100)
plt.show()
print("Moderne Aktivierungsfunktionen gespeichert: moderne_aktivierungen.png")
print("Mish und GELU zeigen oft bessere Performance als ReLU in tiefen Netzen.")

# Beispiel: Mish in einem Keras-Modell
modell_mish = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape=(2,)),
    Mish(),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
modell_mish.compile(optimizer='adam', loss='binary_crossentropy')
print("\nMish-Modell erfolgreich erstellt!")
modell_mish.summary()


# ## Exercise 2
# 
# **Dataset Used:** Custom/Synthetic Array Data (numpy)
# 
# The following code implements the steps for this exercise. Outputs and charts are generated automatically inline.

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
# ============================================================
# educx GmbH – Neuronale Netze | Modul 2
# Lerntag 4: Aktivierungsfunktionen
# Niveau: Experten
# Aufgabe 2 von 3
# ============================================================
# Musterlösung – lauffähig in Spyder (tf_arm conda env)
# Python-Pfad: /Users/solusprime/opt/anaconda3/envs/tf_arm/bin/python
# ============================================================

# Explore information-theoretic properties of activation functions
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

def informationsgehalt_analyse(aktivierung_fn, name, n_proben=10000):
    """Analysiert statistische Eigenschaften der Aktivierungsausgaben"""
    X = np.random.randn(n_proben)
    ausgaben = aktivierung_fn(X)

    # Differentielle Entropie via Histogramm
    hist, bin_edges = np.histogram(ausgaben, bins=50, density=True)
    bin_breite = bin_edges[1] - bin_edges[0]
    hist_pos = hist[hist > 0]
    entropie = -np.sum(hist_pos * np.log(hist_pos + 1e-15) * bin_breite)

    return {
        'ausgaben': ausgaben,
        'entropie': entropie,
        'mittelwert': ausgaben.mean(),
        'std': ausgaben.std(),
        'schiefe': stats.skew(ausgaben)
    }

# Aktivierungsfunktionen definieren
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-500,500)))
def relu(x): return np.maximum(0, x)
def elu(x): return np.where(x > 0, x, 0.5*(np.exp(np.clip(x,-500,0))-1))

funktionen = {
    'Sigmoid': sigmoid,
    'Tanh':    np.tanh,
    'ReLU':    relu,
    'ELU':     elu
}

analysen = {name: informationsgehalt_analyse(fn, name) for name, fn in funktionen.items()}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Informationstheoretische Analyse der Aktivierungen', fontsize=14)

for ax, (name, data) in zip(axes.flatten(), analysen.items()):
    ax.hist(data['ausgaben'], bins=80, alpha=0.7, color='steelblue', density=True, edgecolor='none')
    ax.set_title(f'{name}\nEntropie={data["entropie"]:.3f}, μ={data["mittelwert"]:.3f}, σ={data["std"]:.3f}')
    ax.set_xlabel('Ausgabewert'); ax.set_ylabel('Dichte')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('informationsgehalt.png', dpi=100)
plt.show()

print("=== Aktivierungsstatistiken ===")
for name, data in analysen.items():
    print(f"{name:10s}: Entropie={data['entropie']:.3f} | "
          f"μ={data['mittelwert']:+.3f} | σ={data['std']:.3f} | "
          f"Schiefe={data['schiefe']:+.3f}")
print("\nGespeichert: informationsgehalt.png")


# ## Exercise 3
# 
# **Dataset Used:** Synthetic Classification Data (sklearn.datasets)
# 
# The following code implements the steps for this exercise. Outputs and charts are generated automatically inline.

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
# ============================================================
# educx GmbH – Neuronale Netze | Modul 2
# Lerntag 4: Aktivierungsfunktionen
# Niveau: Experten
# Aufgabe 3 von 3
# ============================================================
# Musterlösung – lauffähig in Spyder (tf_arm conda env)
# Python-Pfad: /Users/solusprime/opt/anaconda3/envs/tf_arm/bin/python
# ============================================================

# Analyze effect of Batch Normalization combined with different activations
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42); np.random.seed(42)

X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)

def erstelle_modell(aktivierung, mit_batchnorm):
    """Modell mit optionaler BatchNormalization"""
    schichten = [tf.keras.layers.Dense(128, input_shape=(X_train.shape[1],))]
    if mit_batchnorm:
        schichten.append(tf.keras.layers.BatchNormalization())
    schichten.append(tf.keras.layers.Activation(aktivierung))

    for _ in range(3):
        schichten.append(tf.keras.layers.Dense(64))
        if mit_batchnorm:
            schichten.append(tf.keras.layers.BatchNormalization())
        schichten.append(tf.keras.layers.Activation(aktivierung))

    schichten.append(tf.keras.layers.Dense(1, activation='sigmoid'))
    m = tf.keras.Sequential(schichten)
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

ergebnisse = {}
for akt in ['relu', 'sigmoid']:
    for bn in [False, True]:
        key = f"{akt} + BN" if bn else akt
        tf.random.set_seed(42)
        m = erstelle_modell(akt, bn)
        h = m.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=50, batch_size=64, verbose=0)
        ergebnisse[key] = h.history
        print(f"{key:20s}: finale Testgenauigkeit = {h.history['val_accuracy'][-1]:.2%}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for key, hist in ergebnisse.items():
    stil = '-' if 'BN' in key else '--'
    axes[0].plot(hist['val_loss'], label=key, linestyle=stil, linewidth=2)
    axes[1].plot(hist['val_accuracy'], label=key, linestyle=stil, linewidth=2)

for ax, titel in zip(axes, ['Validierungsverlust', 'Validierungsgenauigkeit']):
    ax.set_title(titel); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.suptitle('BatchNorm + Aktivierung: Einfluss auf Training', fontsize=13)
plt.tight_layout()
plt.savefig('batchnorm_aktivierung.png', dpi=100)
plt.show()
print("BatchNorm + Aktivierung gespeichert: batchnorm_aktivierung.png")

