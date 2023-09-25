import os
from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from dotenv import load_dotenv
from ml.knn import KnnModel
from ml.svc import SvcModel
from ml.tfidf import TfidfModel
from ml.word2vec import Word2VecModel
from ml.dov2vec import Doc2VecModel
import pandas as pd
import datetime

from ml.preprocessing.preProcessing1 import PreProcessing1
from ml.preprocessing.preProcessing2 import PreProcessing2

load_dotenv()

model_name = os.getenv('MODEL_NAME')
preprocessing_name = os.getenv('PREPROCESSING_NAME')

preprocessings = {
    'preprocessing1': PreProcessing1(),
    'preprocessing2': PreProcessing2()
    # add other preprocessings here
}
preprocessing = preprocessings[preprocessing_name]

# Load CSV into DataFrame
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(
    current_dir, 'scrapping', 'mental_wellness_resources.csv')
resources = pd.read_csv(csv_file_path)
resources = resources.dropna(subset=['description'])

models = {'knn': KnnModel(), 'svc': SvcModel(), 'tfidf': TfidfModel(),
          'word2vec': Word2VecModel(), 'dov2vec': Doc2VecModel()}
model = models[model_name]

trained_model_paths = []
models_trained = []

routes = Blueprint('routes', __name__)


@routes.route('/', methods=['GET', 'POST'])
def index():
    # if request.method == 'POST':
    #     user_input = request.form['user_input']
    #     recommendations = model.recommend(user_input)
    #     print(recommendations)
    #     return render_template('result.html', recommendations=recommendations)

    # return render_template('index.html')

    message = request.args.get('message')
    return render_template('index.html', message=message)


@routes.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['user_input']
    all_recommendations = {}
    global trained_model_paths
    global models_trained

    print(trained_model_paths)
    print("######################")
    print(models_trained)

    for pkl_path in trained_model_paths:
        pkl_filename = os.path.basename(pkl_path)
        model_name, pre_name, _ = pkl_filename.rsplit(
            "_", 2)

        print("=============================")
        print(pkl_path)
        print("=============================")
        print(model_name)
        print("=============================")
        print(pre_name)

        model_instance = models[model_name]
        recommendations = model_instance.recommend(
            pkl_path=pkl_path, user_input=user_input)
        all_recommendations[f"{model_name}_{pre_name}"] = recommendations

    return render_template('result.html', all_recommendations=all_recommendations, models_trained=models_trained, multi=len(models_trained) > 1)


@routes.route('/train', methods=['POST'])
def train():
    # This will store model results
    global trained_model_paths
    trained_model_paths.clear()  # Clear previously stored paths

    global models_trained
    models_trained.clear()

    # Check how many model configurations were provided
    for i in range(1, 6):
        model_key = f"model{i}"
        pre_key = f"preprocessing{i}"

        if model_key not in request.form:
            break

        selected_model = request.form[model_key]
        selected_preprocessing = request.form[pre_key]

        model_instance = models[selected_model]
        global resources
        if selected_preprocessing != 'none':
            preprocessing = preprocessings[selected_preprocessing]
            resources = preprocessing.preProcess(resources)

        # Save model with unique name
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        pkl_filename = f"{selected_model}_{selected_preprocessing}_{timestamp}.pkl"
        pkl_path = os.path.join(current_dir, 'ml', 'models_pkl', pkl_filename)
        model_instance.train(pkl_path=pkl_path, resources_df=resources)
        # Store the path to the trained model
        trained_model_paths.append(pkl_path)

        models_trained.append({
            'model': selected_model,
            'preprocessing': selected_preprocessing,
            'pkl_file': pkl_filename,
            'pkl_path': pkl_path
        })
        print(models_trained)

    if len(models_trained) > 1:
        return render_template('result.html', models_trained=models_trained)
    elif len(models_trained) == 1:
        return render_template('result.html', models_trained=models_trained)
    else:
        return render_template('index.html', message="Aucun modèle n'a été configuré.")
