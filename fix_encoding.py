import json
import os

paths = [
    r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_06_Keras_TensorFlow\notebooks\Anfaenger.ipynb',
    r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_06_Keras_TensorFlow\notebooks\Experte.ipynb',
    r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_06_Keras_TensorFlow\notebooks\Fortgeschrittene.ipynb'
]

for p in paths:
    # Read with default encoding (might be utf-16 or mbcs if powershell screwed it up)
    with open(p, 'rb') as f:
        raw_data = f.read()
    
    # Try decoding
    text = None
    for enc in ['utf-8', 'utf-16', 'windows-1252']:
        try:
            text = raw_data.decode(enc)
            # test if valid json
            json.loads(text)
            print(f'Successfully decoded {os.path.basename(p)} with {enc}')
            break
        except Exception:
            pass
            
    if text:
        # Write back explicitly as utf-8 (no BOM)
        with open(p, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'Saved {os.path.basename(p)} as utf-8')
