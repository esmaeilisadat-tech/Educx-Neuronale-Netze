import subprocess
import os
import sys

folders = ['Tag_01_Grundlagen', 'Tag_02_Perzeptron', 'Tag_03_MLP']
base_dir = r'C:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace'

for folder in folders:
    for diff in ['Anfaenger', 'Fortgeschrittene', 'Experte']:
        nb_path = os.path.join(base_dir, folder, 'notebooks', f'{diff}.ipynb')
        if os.path.exists(nb_path):
            print(f'Executing {nb_path}...')
            res = subprocess.run([sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', nb_path], capture_output=True, text=True)
            if res.returncode != 0:
                print(f'Error executing {nb_path}:')
                print(res.stderr)
            else:
                print(f'Successfully executed {nb_path}')
        else:
            print(f'Missing: {nb_path}')
