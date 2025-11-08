import os
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, send_file
from PIL import Image
import numpy as np
from skimage import measure
import trimesh

app = Flask(__name__)

# Configurer les dossiers d'upload et de résultat
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MODEL_FILE'] = 'static/Segmentation_8.obj'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Charger les modèles 2D et 3D
model_path_2d = os.path.join(os.getcwd(), 'static/models/mon_modele.keras')
print(f"Chargement du modèle 2D à partir de : {model_path_2d}")
model_2d = tf.keras.models.load_model(model_path_2d)

model_path_3d = os.path.join(os.getcwd(), 'static/models/3d_unet_model.h5')
print(f"Chargement du modèle 3D à partir de : {model_path_3d}")
model_3d = tf.keras.models.load_model(model_path_3d)

# Fonctions de traitement
def process_2d_image(file_path, model):
    """Traitement des images 2D pour la détection de tumeurs."""
    image = Image.open(file_path).convert('L').resize((128, 128))
    image = np.expand_dims(np.expand_dims(np.array(image) / 255.0, axis=-1), axis=0)
    prediction = model.predict(image).squeeze()
    result_path = os.path.join(app.config['RESULT_FOLDER'], 'pred_' + os.path.basename(file_path))
    Image.fromarray((prediction * 255).astype(np.uint8)).save(result_path)
    return result_path

def gompertz_predict(V0, r, k, t):
    """Modèle Gompertzien pour prédire la croissance tumorale."""
    return V0 * np.exp((r / k) * (1 - np.exp(-k * t)))

def process_images(file_paths, model):
    """Traitement des images pour la détection 3D et la prédiction de croissance tumorale."""
    try:
        volumes = []
        for file_path in file_paths:
            image = tf.keras.utils.load_img(file_path, color_mode='grayscale', target_size=(128, 128))
            image = tf.keras.utils.img_to_array(image) / 255.0
            volumes.append(image)

        while len(volumes) < 16:
            volumes.append(np.zeros_like(volumes[-1]))
        volumes = volumes[:16]

        volume = np.stack(volumes, axis=0)
        volume = np.moveaxis(volume, 0, -2)
        volume = np.expand_dims(volume, axis=-1)
        volume = np.expand_dims(volume, axis=0)

        predictions = model.predict(volume)
        segmentation = (predictions[0, ..., 0] > 0.5).astype(np.uint8)

        voxel_volume = np.sum(segmentation)
        voxel_size = 1.0
        tumor_volume = voxel_volume * voxel_size

        verts, faces, _, _ = measure.marching_cubes(segmentation, level=0)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'tumor_model.stl')
        mesh.export(result_path)

        r, k, t = 0.2, 0.1, 15
        predicted_volume = gompertz_predict(tumor_volume, r, k, t)

        return tumor_volume, predicted_volume, result_path
    except Exception as e:
        print("Erreur lors du traitement des images :", str(e))
        return None, None, None

# Routes Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index-2', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            return redirect(url_for('display_result', filename=file.filename))
    return render_template('index-2.html')

@app.route('/result2/<filename>')
def display_result(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    prediction_path = process_2d_image(file_path, model_2d)
    return render_template('result2.html', original=filename, prediction=os.path.basename(prediction_path))

@app.route('/index-3', methods=['GET', 'POST'])
def index3():
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        if files:
            for file in files:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            return render_template('result3.html')
    return render_template('index-3.html')

@app.route('/index-4', methods=['GET', 'POST'])
def index4():
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        if files:
            file_paths = [os.path.join(app.config['UPLOAD_FOLDER'], file.filename) for file in files]
            for file, path in zip(files, file_paths):
                file.save(path)

            tumor_volume, predicted_volume, result_path = process_images(file_paths, model_3d)
            if tumor_volume is None:
                return "Erreur lors du traitement des images", 500

            return render_template('result4.html', tumor_volume=tumor_volume, predicted_volume=predicted_volume, result_path=result_path)
    return render_template('index-4.html')

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['RESULT_FOLDER'], filename)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
