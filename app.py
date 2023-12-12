from flask import Flask, request, render_template
import pandas as pd 
import pickle as pkl
from sklearn.preprocessing import StandardScaler

scaler = pkl.load(open('models/scaler.pkl','rb'))
regressor = pkl.load(open('models/regressor.pkl','rb'))
tree = pkl.load(open('models/tree.pkl','rb'))
naive = pkl.load(open('models/naive.pkl','rb'))



app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit',methods = ['GET','POST'])
def submit():
    if request.method=='POST':
        Pregnancies = float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
        data = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
        data_scaled = scaler.transform([data])
        result_lr = regressor.predict(data_scaled)
        result_tree = tree.predict([data])
        result_naive = naive.predict(data_scaled)
        count_1 = 0
        count_0 = 0
        if(result_lr[0])==0:
            count_0 = count_0 + 1
        else:
            count_1 = count_1 + 1

        if(result_tree[0])==0:
            count_0 = count_0 + 1
        else:
            count_1 = count_1 + 1

        if(result_naive[0])==0:
            count_0 = count_0 + 1
        else:
            count_1 = count_1 + 1
        if(count_1>count_0):
            return render_template('index.html',result="1",result_lr=result_lr[0],result_tree=result_tree[0],result_naive=result_naive[0])
        else:
            return render_template('index.html',result="0")

        
        

        #return "Your prediction is {}".format(result
    else:
        return render_template('index.html')

    

if __name__ == "__main__":
    app.run(debug=True)