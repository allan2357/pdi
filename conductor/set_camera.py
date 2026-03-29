import json

nb_path = r'C:\Users\allan\Downloads\pdi\Projeto\DogFlush_Entregavel.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        full_code = "".join(source)
        
        target = '    if not cap.isOpened():\n        print(f"Erro ao abrir a fonte de vídeo: {modo_video}")\n        return\n'
        replacement = target + '\n    # Padronizando a câmera: Resolução (640x480) e FPS (15)\n    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)\n    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)\n    cap.set(cv2.CAP_PROP_FPS, 15)\n'
        
        if target in full_code:
            full_code = full_code.replace(target, replacement)
            
            # Reconstruct the source list properly
            lines = full_code.split('\n')
            new_source = [line + '\n' for line in lines[:-1]]
            if not full_code.endswith('\n'):
                new_source[-1] = new_source[-1][:-1] # remove the extra \n we just added to the last element
            
            cell['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Câmera padronizada com sucesso no notebook.")
