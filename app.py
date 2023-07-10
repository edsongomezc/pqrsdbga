from flask import Flask, render_template, request
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.request import urlretrieve
import os

app = Flask(__name__)

# URLs de los modelos y archivos tfidf en Google Cloud Storage
model_urls = [
    "https://storage.googleapis.com/pqrsdbga/primer_componente_final.joblib",
    "https://storage.googleapis.com/pqrsdbga/segundo_componente_final.joblib",
    "https://storage.googleapis.com/pqrsdbga/tercer_componente_final.joblib",
    "https://storage.googleapis.com/pqrsdbga/cuarto_componente_final.joblib",
    "https://storage.googleapis.com/pqrsdbga/tfidf_primer_componente_final.joblib",
    "https://storage.googleapis.com/pqrsdbga/tfidf_segundo_componente_final.joblib",
    "https://storage.googleapis.com/pqrsdbga/tfidf_tercer_componente_final.joblib",
    "https://storage.googleapis.com/pqrsdbga/tfidf_cuarto_componente_final.joblib",
]

# Nombres de archivo locales para los modelos y archivos tfidf
model_filenames = [
    "primer_componente_final.joblib",
    "segundo_componente_final.joblib",
    "tercer_componente_final.joblib",
    "cuarto_componente_final.joblib",
    "tfidf_primer_componente_final.joblib",
    "tfidf_segundo_componente_final.joblib",
    "tfidf_tercer_componente_final.joblib",
    "tfidf_cuarto_componente_final.joblib",
]

# Descargar los modelos y archivos tfidf si no existen
for url, filename in zip(model_urls, model_filenames):
    if not os.path.exists(filename):
        urlretrieve(url, filename)

# Cargar los modelos
componente1 = load(model_filenames[0])
componente2 = load(model_filenames[1])
componente3 = load(model_filenames[2])
componente4 = load(model_filenames[3])

# Cargar los archivos tfidf
tfidf1 = load(model_filenames[4])
tfidf2 = load(model_filenames[5])
tfidf3 = load(model_filenames[6])
tfidf4 = load(model_filenames[7])

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
