"""
Módulo VModel: Proporciona una clase para trabajar con embeddings de texto y buscar coincidencias en una base de datos de tags.
@author: Martinez, Nicolas Agustin
"""

import json
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset

class VModel:
    def __init__(self, tags_path):
        """
        Inicializa la instancia VModel.
        
        Args:
        - tags_path (str): Ruta al archivo JSON que contiene los tags.
        """
        self.tags_path = tags_path

        # Inicializar el modelo y el tokenizador de transformers
        model_ckpt = 'all-MiniLM-L6-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt)

        # Cargar datos de tags desde el archivo JSON
        with open(tags_path, "r", encoding='utf-8') as file:
            self.tags_data = json.load(file)
            file.flush()

        # Extraer todos los tags y preparar el contexto
        all_tags = [tags for tags in self.tags_data.values()]
        self.context = {'context': all_tags}

        # Convertir los datos de tags en un formato adecuado para el dataset
        tag_ids = list(self.tags_data.keys())
        tag_texts = [" ".join(tags) for tags in self.tags_data.values()]

        # Crear un dataset con los datos extraídos del archivo tags.json
        self.tags_dataset = Dataset.from_dict({"id": tag_ids, "context": tag_texts})
        self.context_dataset = Dataset.from_dict(self.context)

        # Cargar el modelo y preparar el índice FAISS
        self.load_model()

    def mean_pooling(self, model_output, attention_mask):
        """
        Realiza mean pooling en los embeddings para obtener una representación consolidada.
        
        Args:
        - model_output (torch.Tensor): Salida del modelo.
        - attention_mask (torch.Tensor): Máscara de atención para el input.
        
        Returns:
        - torch.Tensor: Embedding consolidado.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, text_list):
        """
        Obtiene embeddings para una lista de textos utilizando el modelo y el tokenizador.
        
        Args:
        - text_list (list): Lista de textos para los cuales obtener embeddings.
        
        Returns:
        - torch.Tensor: Embeddings de los textos.
        """
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return self.mean_pooling(model_output, encoded_input["attention_mask"])

    def find_best_matches(self, user_input, k=10):
        """
        Busca las k mejores coincidencias para el input del usuario en el dataset de tags.
        
        Args:
        - user_input (str): Texto de entrada del usuario.
        - k (int): Número de coincidencias a retornar.
        
        Returns:
        - list: Lista de las k mejores coincidencias.
        """
        print('[find_best_matches] -- BUSCANDO MEJORES COINCIDENCIAS --')
        user_embedding = self.get_embeddings([user_input]).cpu().detach().numpy()
        scores, samples = self.context_dataset.get_nearest_examples("embeddings", user_embedding, k=k)
        return samples

    def load_model(self):
        """
        Vectoriza la base de datos de tags y agrega un índice FAISS para búsquedas rápidas.
        """
        print('[load_model] -- INICIANDO VECTORIZACION DE LA BASE DE DATOS -- ')
        self.context_dataset = self.context_dataset.map(lambda x: {"embeddings": self.get_embeddings(x["context"]).cpu().numpy()[0]})
        self.context_dataset.add_faiss_index(column="embeddings")

    def get_simils(self, sentence):
        """
        Obtiene tags similares para una oración dada.
        
        Args:
        - sentence (str): Oración para la cual buscar tags similares.
        
        Returns:
        - list: Lista de tags similares.
        """
        results = self.find_best_matches(sentence)
        matches = []
        for k, v in self.tags_data.items():
            for r in results['context']:
                if r in v:
                    match = {'id': k, 'match': v[0], 'tags': v}
                    if match not in matches:
                        matches.append(match)
        return matches

    def get_by_ID(self, id):
        """
        Obtiene tags por ID. Función aún no implementada.
        
        Args:
        - id (str): ID del tag a buscar.
        
        Returns:
        - list: Datos del tag o un diccionario vacío si no se encuentra.
        """
        print('[GET BY ID]')
        if self.tags_data.get(id):
            return [{'id': id, 'match': self.tags_data[id][0]}]
        return [{}]
