from flask import Flask, render_template,request
from flask_sqlalchemy import SQLAlchemy
# for model prediction
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

import keras

#@title df     -  ( DataFrame )
breast_cancer = load_breast_cancer()
# cnvert into dataframe
x_train = pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)



# -----------------------------------------------------------------app
app = Flask(__name__,template_folder='templates')
# -------------------------------------------------------------------DataBase
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqllite:///Database/db.sqllite3'



# Load model
model = keras.models.load_model('./static/breast_cancer_model/breast_cancer_ann_model1.h5')

# Get data and give the result function
def get_give(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30):
    # convert data into dataframe and scaled
    lst = [[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30]]
    arr = np.array(lst)
    data = pd.DataFrame(np.array(lst),columns=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension'])

    sc = StandardScaler()
    sc.fit(x_train)
    x_data_sc = sc.transform(data)
    pred = model.predict(x_data_sc)
    # pred = np.array([[0,2,3,4,5,6,7]])#temp
    pred_value = pred[0][0]

    return pred_value


# ------------------------------------------------------------render template
@app.route('/',methods=['GET','POST'])
def index():
    ggplot = 'Result will show here.'
    if (request.method=='POST'):
        f1 = request.form['f1']
        f2 = request.form['f2']
        f3 = request.form['f3']
        f4 = request.form['f4']
        f5 = request.form['f5']
        f6 = request.form['f6']
        f7 = request.form['f7']
        f8 = request.form['f8']
        f9 = request.form['f9']
        f10 = request.form['f10']
        f11 = request.form['f11']
        f12 = request.form['f12']
        f13 = request.form['f13']
        f14 = request.form['f14']
        f15 = request.form['f15']
        f16 = request.form['f16']
        f17 = request.form['f17']
        f18 = request.form['f18']
        f19 = request.form['f19']
        f20 = request.form['f20']
        f21 = request.form['f21']
        f22 = request.form['f22']
        f23 = request.form['f23']
        f24 = request.form['f24']
        f25 = request.form['f25']
        f26 = request.form['f26']
        f27 = request.form['f27']
        f28 = request.form['f28']
        f29 = request.form['f29']
        f30 = request.form['f30']
        pred = get_give(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30)
        
        if pred >= 0.5:
            res_val = "According to Data-: You have 'Breast Cancer'"
            ggplot=res_val
        elif pred < 0.5:
            res_val = "According to Data-: You have 'Not Breast Cancer'"
            ggplot=res_val
        else:
            res_val = "Sorry ! Fill the correct value"
            ggplot=res_val

    return render_template('index.html',pred=ggplot)

@app.route('/readme')
def readme():
    return render_template('readmi.html')







if __name__=='__main__':
    app.run(debug=False)