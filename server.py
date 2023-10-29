from flask import Flask, request, render_template 
from Conversion_model import calculate_frequencies

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/freq", methods=["POST"])
def freq():
    emotion_data = request.get_json()
    if len(emotion_data) != 0:
        outputs = calculate_frequencies(emotion_data).tolist()
    else:
        outputs = "No face detected"
    return outputs


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=3000, ssl_context="adhoc")