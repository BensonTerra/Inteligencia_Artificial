import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

app = Flask(__name__)

#import the model and the clf
model=pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    flt_features = [float(x) for x in request.form.values()]
    final_features = [np.array(flt_features)]
    #final_features = clf.transform(final_features)
    
    prediction = model.predict(final_features)
    
    print(prediction)
    
    output =  round(prediction[0],2)
    
    return render_template('index.html', prediction_text=f'Estimated value: {output:,}')
    

@app.route('/results', methods=['POST'])
def results():
    features=request.get_json(force=True)
    final_features = [np.array(list(features.values()))]
    #final_features = clf.transform(final_features)
  
    prediction = model.predict(final_features)
    
    output =  round(prediction[0],2)
    
    return jsonify(output)
    

if __name__ == "__main__":
    app.run(debug=True)
