from flask import Flask, render_template, request
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.request import urlretrieve
import os
import tempfile

app = Flask(__name__)

# URLs de los modelos y archivos tfidf en Google Cloud Storage
model_urls = [
    "https://storage.googleapis.com/pqrsdbga.appspot.com/primer_componente_final.joblib",
    "https://storage.googleapis.com/pqrsdbga.appspot.com/segundo_componente_final.joblib",
    "https://storage.googleapis.com/pqrsdbga.appspot.com/tercer_componente_final.joblib",
    "https://storage.googleapis.com/pqrsdbga.appspot.com/cuarto_componente_final.joblib",
    "https://storage.googleapis.com/pqrsdbga.appspot.com/tfidf_primer_componente_final.joblib",
    "https://storage.googleapis.com/pqrsdbga.appspot.com/tfidf_segundo_componente_final.joblib",
    "https://storage.googleapis.com/pqrsdbga.appspot.com/tfidf_tercer_componente_final.joblib",
    "https://storage.googleapis.com/pqrsdbga.appspot.com/tfidf_cuarto_componente_final.joblib",
]

# Descargar los modelos y archivos tfidf si no existen y cargarlos
models = []
for url in model_urls:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    urlretrieve(url, temp_file.name)
    models.append(load(temp_file.name))

# Asignar modelos y archivos tfidf a variables
componente1, componente2, componente3, componente4, tfidf1, tfidf2, tfidf3, tfidf4 = models

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']

        # Transformar el texto con cada uno de los vectorizadores
        x1 = tfidf1.transform([text])
        x2 = tfidf2.transform([text])
        x3 = tfidf3.transform([text])
        x4 = tfidf4.transform([text])
        
        # Realizar las predicciones con cada uno de los modelos
        result1 = componente1.predict(x1)
        result2 = componente2.predict(x2) if result1 == 'otro' else result1
        result3 = componente3.predict(x3) if result2 == 'Extra' else result2
        result4 = componente4.predict(x4) if result3 == 'Extra' else result3

        return result4[0]
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
