from flask import Flask, render_template, request, jsonify
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)

# Load your model
model = load_model('static/quantized_model.h5')
def load_tokenizer_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Example usage:
tokenizer_path = 'static/tokenizer.pkl'  # Replace this with the path to your tokenizer file
tokenizer = load_tokenizer_from_pickle(tokenizer_path)



# Function to detect plagiarism
def pre_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen = 111)
    return padded_sequences
def prdict_plagiarism(text):
    process_text = pre_text(text)
    predictions = model.predict(process_text)
    return predictions[0][0]
    
def detect_plagiarism(text):
    plagiarism_score = prdict_plagiarism(text)
    if plagiarism_score > 0.5:
        string =  f'This Text Has Plagiarism With Similarity Score IS: {plagiarism_score}'
    else:
        string = 'This Text Has No Plagiarism'
    print(string)
    return string


@app.route('/plagiarism')
def plagiarism():
    return render_template('plagiarism.html')
@app.route('/')
def home():
   return render_template("Home.html")
@app.route('/login')
def login():
   return render_template("login.html")
@app.route('/signup')
def signup():
   return render_template("signup.html")

@app.route('/check_plagiarism', methods=['POST'])
def check_plagiarism():
    text = request.form['text']
    plagiarism_result = detect_plagiarism(text)
    return jsonify({'result': plagiarism_result})

@app.route('/upload_model', methods=['POST'])
def upload_model():
    model_file = request.files['model']
    tokenizer_file = request.files['tokenizer']
    model_path = 'quantized_model.h5'
    tokenizer_path = 'tokenizer.pkl'
    model_file.save(model_path)
    tokenizer_file.save(tokenizer_path)
    load_model_and_tokenizer(model_path, tokenizer_path)
    return jsonify({'message': 'Model and tokenizer uploaded successfully!'})

if __name__ == '__main__':
    app.run(debug=True)
