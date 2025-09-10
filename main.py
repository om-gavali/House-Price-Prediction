import pandas as pd
import pickle
import numpy as np
from flask import Flask, render_template, request

# Load data & model
data = pd.read_csv('clean_data.csv')
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    # Ensure correct column names
    input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                            columns=['location', 'total_sqft', 'bath', 'BHK'])

    prediction = pipe.predict(input_df)[0] * 1e5

    return str(np.round(prediction, 2))  # rounded to 2 decimal places


if __name__ == '__main__':
    app.run(debug=True, port=5000)
