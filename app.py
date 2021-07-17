import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model/model.pkl','rb'))
@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict',methods=['POST'])
def predict():
    input_features=[float(x) for x in request.form.values()]
    features_value=[np.array(input_features)]
    features_name=["Age","Smokes","AreaQ","Alkhol"]
    data=pd.DataFrame(features_value,columns=features_name)
    output=model.predict(data)

    if output==1:
        res_val="**Lung Cancer**"
    else:
        res_val="**No Lung Cancer**"

    return render_template('main.html',prediction_text="Patient has {}".format(res_val))

if __name__=="__main__":
    app.run(debug=True)
