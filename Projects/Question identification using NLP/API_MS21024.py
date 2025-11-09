from flask import Flask, request
import re
import pickle
import os

app = Flask(__name__)


def preprocess_text(sen):
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


current_dir = os.path.dirname(os.path.abspath(__file__))


count_vectorizer_path = os.path.join(current_dir, "count_vectorizer.pkl")
lda_model_path = os.path.join(current_dir, "lda_model.pkl")
random_forest_model_path = os.path.join(current_dir, "random_forest_model.pkl")


with open(count_vectorizer_path, "rb") as f:
    count_vectorizer = pickle.load(f)

with open(lda_model_path, "rb") as f:
    lda = pickle.load(f)

with open(random_forest_model_path, "rb") as f:
    rf = pickle.load(f)


def predict_sentence_or_question(sentence):
    sentence_transformed = lda.transform(count_vectorizer.transform([preprocess_text(sentence)]))
    prediction = rf.predict(sentence_transformed)
    return "sentence" if prediction == 'sentence' else "question"


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        sentence = request.form.get('sentence')
        prediction = predict_sentence_or_question(sentence)
        html_response = f"""
        <html>
            <head>
                <title>Sentence or Question Prediction</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        text-align: center;
                        padding: 50px;
                    }}
                    h1 {{
                        color: #2C3E50;
                    }}
                    p {{
                        color: #34495E;
                    }}
                    input[type="text"] {{
                        padding: 10px;
                        width: 300px;
                        margin-bottom: 20px;
                    }}
                    input[type="submit"] {{
                        padding: 10px 20px;
                        background-color: #3498db;
                        color: #fff;
                        border: none;
                        cursor: pointer;
                    }}
                </style>
            </head>
            <body>
                <h1>Sentence or Question Prediction</h1>
                <p>The input text is: <strong>{sentence}</strong></p>
                <p><strong>Prediction:</strong> {prediction}</p>
                <form method="post">
                    <input type="text" name="sentence" placeholder="Enter a sentence or question"><br>
                    <input type="submit" value="Predict">
                </form>
            </body>
        </html>
        """
        return html_response
    else:
        return """
        <html>
            <head>
                <title>Sentence or Question Prediction</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        text-align: center;
                        padding: 50px;
                    }}
                    h1 {{
                        color: #2C3E50;
                    }}
                    p {{
                        color: #34495E;
                    }}
                </style>
            </head>
            <body>
                <h1>Sentence or Question Prediction</h1>
                <form method="post">
                    <input type="text" name="sentence" placeholder="Enter a sentence or question"><br>
                    <input type="submit" value="Predict">
                </form>
            </body>
        </html>
        """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)