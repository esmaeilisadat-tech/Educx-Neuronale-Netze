import json

path = r'C:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_03_MLP\notebooks\Experte.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Just a quick check to ensure we can parse it. If not, we will need to rewrite the cell source from scratch or fix the string.
try:
    data = json.loads(text)
    print("JSON is valid!")
except Exception as e:
    print("JSON is invalid:", e)
