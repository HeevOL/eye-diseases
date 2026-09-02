import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import glob
from pathlib import Path


def configurar_gpu():
    """Configura a GPU para uso eficiente (mesma lógica usada no treinamento)"""
    dispositivos_gpu = tf.config.list_physical_devices('GPU')

    if dispositivos_gpu:
        try:
            for dispositivo in dispositivos_gpu:
                tf.config.experimental.set_memory_growth(dispositivo, True)
            print(f"GPU configurada: {len(dispositivos_gpu)} placa(s) encontrada(s)")
        except RuntimeError as erro:
            print(f"Aviso ao configurar GPU: {erro}")
    else:
        print("Nenhuma GPU encontrada. Predição vai rodar na CPU.")


def load_class_indices():
    """Carrega o mapeamento de classes do arquivo salvo"""
    class_indices = {}
    try:
        with open('class_indices.txt', 'r') as f:
            for line in f:
                class_name, index = line.strip().split(':')
                class_indices[class_name] = int(index)
        return class_indices
    except FileNotFoundError:
        print("Arquivo class_indices.txt não encontrado. Execute o treinamento primeiro.")
        return None

def predict_disease(model, img_path, class_indices, img_width=224, img_height=224):
    """
    Função para prever a classe de uma imagem de input.
    Retorna: (classe_predita, probabilidade, categoria_real)
    """
    try:
        img = image.load_img(img_path, target_size=(img_width, img_height))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = model.predict(img_array, verbose=0)
        
        predicted_class_index = np.argmax(predictions[0])
        predicted_probability = predictions[0][predicted_class_index]

        inverted_indices = {v: k for k, v in class_indices.items()}
        predicted_class_name = inverted_indices[predicted_class_index]

        actual_category = Path(img_path).parent.name
        
        is_correct = (actual_category == predicted_class_name)
        
        print(f"\n{'='*60}")
        print(f"Imagem: {os.path.basename(img_path)}")
        print(f"Pasta (Categoria Real): {actual_category}")
        print(f"Predição do Modelo: {predicted_class_name}")
        print(f"Probabilidade: {predicted_probability * 100:.2f}%")
        print(f"Status: {'✅ CORRETO' if is_correct else '❌ ERRADO'}")
        
        print(f"\nProbabilidades para todas as classes:")
        for i, prob in enumerate(predictions[0]):
            class_name = inverted_indices[i]
            indicator = "← PREDITO" if i == predicted_class_index else ""
            print(f"  - {class_name}: {prob * 100:.2f}% {indicator}")
        print(f"{'='*60}")

        return predicted_class_name, predicted_probability, actual_category, is_correct
        
    except Exception as e:
        print(f"Erro ao processar a imagem {img_path}: {e}")
        return None, None, None, None


def predict_all_images_in_folder():
    """Faz predição para todas as imagens na pasta inputs e suas subpastas"""

    configurar_gpu()

    try:
        model = load_model('eye_disease_model.h5')
        print("Modelo carregado com sucesso!")
    except:
        try:
            model = load_model('eye_disease_model_final.h5')
            print("Modelo final carregado com sucesso!")
        except:
            print("Erro: Nenhum modelo encontrado. Execute o treinamento primeiro.")
            return

    class_indices = load_class_indices()
    if class_indices is None:
        return

    input_folder = 'inputs'
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    image_paths = set()
    
    for extension in image_extensions:
        search_pattern = os.path.join(input_folder, '**', extension)
        found_paths = glob.glob(search_pattern, recursive=True)
        
        for path in found_paths:
            image_paths.add(os.path.abspath(path))
        
        search_pattern_upper = os.path.join(input_folder, '**', extension.upper())
        found_paths_upper = glob.glob(search_pattern_upper, recursive=True)
        
        for path in found_paths_upper:
            image_paths.add(os.path.abspath(path))

    image_paths = list(image_paths)
    
    if not image_paths:
        print(f"Nenhuma imagem encontrada na pasta '{input_folder}' e suas subpastas")
        return

    print(f"Encontradas {len(image_paths)} imagens únicas para predição...")
    
    images_by_category = {}
    for img_path in image_paths:
        
        rel_path = os.path.relpath(img_path, input_folder)
        category = os.path.dirname(rel_path)
        if not category:  
            category = 'root'
        
        if category not in images_by_category:
            images_by_category[category] = []
        images_by_category[category].append(img_path)
    
    print(f"\nEstrutura encontrada em '{input_folder}':")
    for category, images in images_by_category.items():
        print(f"  - {category}: {len(images)} imagens")
    
    all_results = []
    print(f"\n{'#'*80}")
    print("INICIANDO PREDIÇÕES...")
    print(f"{'#'*80}")
    
    for category, category_images in images_by_category.items():
        print(f"\n📁 Processando categoria: {category}")
        print(f"📷 Total de imagens: {len(category_images)}")
        
        category_results = []
        for img_path in category_images:
            
            predicted_class, probability, actual_category, is_correct = predict_disease(model, img_path, class_indices)
            
            if predicted_class:
                result = {
                    'image': os.path.basename(img_path),
                    'actual': actual_category,
                    'prediction': predicted_class,
                    'probability': probability,
                    'is_correct': is_correct
                }
                category_results.append(result)
                all_results.append(result)
                
        if category_results and category != 'root':
            correct_predictions = sum(1 for r in category_results if r['is_correct'])
            accuracy = (correct_predictions / len(category_results)) * 100
            print(f"\n📊 Estatísticas para {category}:")
            print(f"   ✅ Corretas: {correct_predictions}/{len(category_results)}")
            print(f"   📈 Acurácia: {accuracy:.2f}%")

    print(f"\n{'#'*80}")
    print("RESUMO GERAL DAS PREDIÇÕES")
    print(f"{'#'*80}")
    
    known_results = [r for r in all_results if r['actual'] != 'root']
    
    if known_results:
        total_correct = sum(1 for r in known_results if r['is_correct'])
        total_images = len(known_results)
        overall_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
        
        print(f"📊 Total de imagens com classe conhecida: {total_images}")
        print(f"✅ Predições corretas: {total_correct}")
        print(f"❌ Predições incorretas: {total_images - total_correct}")
        print(f"🎯 Acurácia geral: {overall_accuracy:.2f}%")
        
        print(f"\n📈 Detalhamento por categoria:")
        categories_summary = {}
        for result in known_results:
            category = result['actual']
            if category not in categories_summary:
                categories_summary[category] = {'total': 0, 'correct': 0}
            categories_summary[category]['total'] += 1
            if result['is_correct']:
                categories_summary[category]['correct'] += 1
        
        for category, stats in categories_summary.items():
            accuracy = (stats['correct'] / stats['total']) * 100
            print(f"   {category}: {stats['correct']}/{stats['total']} ({accuracy:.2f}%)")
        
        incorrect_predictions = [r for r in known_results if not r['is_correct']]
        if incorrect_predictions:
            print(f"\n⚠️  Exemplos de predições incorretas:")
            for i, result in enumerate(incorrect_predictions[:5]): 
                print(f"   {i+1}. {result['image']}:")
                print(f"      Real: {result['actual']} → Predito: {result['prediction']} ({result['probability']*100:.2f}%)")
    
    root_results = [r for r in all_results if r['actual'] == 'root']
    if root_results:
        print(f"\n📋 Imagens da pasta 'inputs/' (classe desconhecida):")
        for result in root_results:
            print(f"   - {result['image']}: {result['prediction']} ({result['probability']*100:.2f}%)")


def predict_single_image(img_path):
    """Faz predição para uma única imagem específica"""

    configurar_gpu()

    try:
        model = load_model('eye_disease_model.h5')
    except:
        try:
            model = load_model('eye_disease_model_final.h5')
        except:
            print("Erro: Nenhum modelo encontrado. Execute o treinamento primeiro.")
            return

    class_indices = load_class_indices()
    if class_indices is None:
        return

    if not os.path.exists(img_path):
        print(f"Erro: Arquivo {img_path} não encontrado!")
        return

    result = predict_disease(model, img_path, class_indices)
    return result

if __name__ == "__main__":
    predict_all_images_in_folder()
    