# Módulo VModel para la Vectorización de Texto y Búsqueda de Coincidencias en Tags

## Autor
Martinez, Nicolas Agustin

## Descripción
El módulo VModel proporciona una clase diseñada para trabajar con embeddings de texto y buscar coincidencias en una base de datos de tags. Utiliza la biblioteca `transformers` para obtener embeddings de texto y la biblioteca `datasets` para gestionar y buscar en el conjunto de datos de tags.

## Funcionalidades Principales
- Inicialización y carga de un modelo preentrenado y su tokenizador.
- Carga de datos de tags desde un archivo JSON.
- Vectorización de la base de datos de tags y creación de un índice FAISS para búsquedas rápidas.
- Obtención de embeddings para una lista de textos.
- Búsqueda de las mejores coincidencias para un texto de entrada en el conjunto de datos de tags.
- (No implementado) Búsqueda de tags por ID.

## Instrucciones de Uso
1. Asegúrese de tener todas las dependencias instaladas. Estas incluyen `torch`, `transformers` y `datasets`.
2. Inicialice una instancia de la clase VModel con la ruta al archivo JSON que contiene los tags.
3. Utilice el método `get_simils(sentence)` para obtener tags similares para una oración dada.
4. (Opcional) Si necesita buscar tags por ID, utilice el método `get_by_ID(id)`.

## Ejemplo
```python
vm = VModel(tags_path="ruta/a/tags.json")
resultados = vm.get_simils("Ejemplo de oración")
print(resultados)
```

> **Notas:** 
> - El método get_by_ID(id) aún no está implementado.
> - Asegúrese de tener el archivo tags.json en la ruta especificada o proporcione la ruta correcta al inicializar la clase VModel.

---

# API REST en Flask

## Descripción
Este es un servicio API REST desarrollado en Flask. Si estás utilizando Windows, es necesario instalar waitress para servir la aplicación. Puedes hacerlo con el comando `pip install waitress` y luego ejecutar la aplicación con `waitress-serve --port=3000 app:app`.

## Dependencias
- Flask
- flask_cors
- waitress (solo para Windows)
- vector_model (módulo personalizado)

## Inicialización
Se carga el archivo tags.json a través del modelo VModel y se inicializa la aplicación Flask.

## Endpoints
### 1. /vectorDB (Método POST)
**Descripción:** Devuelve similitudes basadas en el texto proporcionado.  
**Entrada:** JSON con el campo text.  
**Salida:** JSON con las similitudes encontradas.  
**Errores:** Si ocurre un error durante la ejecución, se imprime el error y se devuelve un objeto JSON vacío.
