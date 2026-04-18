from flask import Flask, request, render_template
import numpy as np
from model import train_model

app = Flask(__name__)

model = train_model()

@app.route('/')
def home():
    return '''
    <h2>Transformer Health Prediction</h2>
    <form method="post" action="/predict">
    Oil Temp: <input name="oil"><br>
    Winding Temp: <input name="wind"><br>
    Load Current: <input name="load"><br>
    Voltage: <input name="volt"><br>
    Ambient Temp: <input name="amb"><br>
    Moisture: <input name="moist"><br>
    H2: <input name="h2"><br>
    CO: <input name="co"><br>
    CH4: <input name="ch4"><br>
    <input type="submit" value="Predict">
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    values = [float(x) for x in request.form.values()]
    final = [np.array(values)]
    prediction = model.predict(final)

    if prediction[0] == "Healthy":
       return f"<h3 style='color:green;'>Health Status: {prediction[0]}</h3>"
    elif prediction[0] == "Warning":
        return f"<h3 style='color:orange;'>Health Status: {prediction[0]}</h3>"
    else:
        return f"<h3 style='color:red;'>Health Status: {prediction[0]}</h3>"

if __name__ == "__main__":
    app.run(debug=True)