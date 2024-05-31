from flask import Blueprint, render_template, request
from controller import classify_sequence, get_model_accuracy

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/classify', methods=['POST'])
def classify():
    sequence = request.form['sequence']
    prediction, accuracy = classify_sequence(sequence)
    return render_template('result.html', sequence=sequence, prediction=prediction, accuracy=accuracy)
