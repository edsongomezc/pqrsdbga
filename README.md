# PROYECTO DE GRADO - PROTOTIPO DE HERRAMIENTA PARA LA MEJORA EN LOS PROCESOS DE DESIGNACIÓN DE PQRSD DE LA ALCALDIA DE BUCARAMANGA

Cómo parte de los objetivos del proyecto, se desarrolla un prototipo o herramienta web y móvil  a través del framework Flask que permite a los usuarios funcionales de la Alcaldía de Bucaramanga minimizar los tiempos de designación y tener recomendaciones o sugerencias de áreas a designar las PQRSD que se reciban en la entidad.

La aplicación web utiliza los modelos de clasificación previamente entrenados para realizar las predicciones sobre los textos ingresados. Estos modelos serán cargados y utilizados en conjunto con técnicas de vectorización, como TF-IDF, para convertir los textos en representaciones numéricas adecuadas para el procesamiento. El sistema permite a los usuarios ingresar sus textos a través de un formulario en la interfaz web. Una vez que se envíe el texto, se realizará una solicitud POST al servidor Flask, donde se realizará la transformación del texto y se ejecutará el modelo de clasificación correspondiente para obtener la predicción. 

Además, se implementó un diseño visual atractivo y responsive para pantallas móviles utilizando HTML, CSS y librerías de estilos como Bootstrap. Se incluye la funcionalidad de modales para mostrar las predicciones en una ventana emergente sin necesidad de recargar la página.

## Librerías

blinker==1.6.2
click==8.1.4
Flask==2.3.2
gunicorn==20.1.0
importlib-metadata==6.8.0
itsdangerous==2.1.2
Jinja2==3.1.2
joblib==1.3.1
MarkupSafe==2.1.3
scikit-learn==1.3.0
Werkzeug==2.3.6
zipp==3.16.0

Ver, requirements.txt

![image](https://github.com/edsongomezc/pqrsdbga/assets/7485878/8a1fe3ca-9689-458d-8cd8-adef447d4776)
