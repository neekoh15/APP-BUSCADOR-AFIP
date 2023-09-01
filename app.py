"""
API REST en Flask
Instrucciones para Windows:
1. Instalar waitress: pip install waitress
2. Ejecutar la aplicación: waitress-serve --port=3000 app:app

@autor: Martinez, Nicolas Agustin
"""

# Importación de módulos necesarios

from flask import Flask, request, jsonify
from flask_cors import CORS
from vector_model import VModel


# Inicialización del modelo VModel y carga del archivo tags.json
vectorDB = VModel(tags_path='tags.json')

# Inicialización de la aplicación Flask
app = Flask(__name__)

# Habilitación de CORS para la aplicación
CORS(app)

# (Opcional) Para montar la API en línea, descomentar la siguiente línea
# run_with_ngrok(app)

@app.route('/vectorDB', methods=['POST'])
def get_simils():
    """
    Endpoint para obtener similitudes basadas en el texto proporcionado.
    
    Entrada: JSON con el campo 'text'
    Salida: JSON con las similitudes encontradas o un objeto JSON vacío en caso de error
    """
    
    # Obtener datos del cuerpo de la solicitud
    data:dict = request.json
    
    # Si no hay datos, retornar None
    if not data:
        return None
    
    # Extraer el texto del JSON
    text:str = data['text']

    # Intentar obtener similitudes y devolver la respuesta
    try:
        response:dict = vectorDB.get_simils(text)
        return jsonify(response)
    except Exception as e:
        # Imprimir el error y devolver un objeto JSON vacío
        print('Error al obtener similitudes:', e)
        return {}
