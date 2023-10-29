from flask import Flask, request, render_template 

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/freq")
def freq():
    return request.body


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=3000)