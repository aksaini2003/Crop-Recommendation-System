from flask import Flask,render_template,url_for,request,jsonify
import joblib
import os
scaler=joblib.load('standardscaler.pkl','rb')
kmeans=joblib.load('model.pkl','rb')
import pickle 
dt=pickle.load(open('labeled_output.pkl','rb'))

# Crop Image Mapping
CROP_IMAGES = {
    'Rice': 'rice.jpg',
    'Maize': 'maize.jpg',
    'Jute': 'jute.jpg',
    'Cotton': 'cotton.jpg',
    'Coconut': 'coconut.jpg',
    'Papaya': 'papaya.jpg',
    'Orange': 'orange.jpg',
    'Apple': 'apple.jpg',
    'Muskmelon': 'muskmelon.jpg',
    'Watermelon': 'watermelon.jpg',
    'Grapes': 'grapes.jpg',
    'Mango': 'mango.jpg',
    'Banana': 'banana.jpg',
    'Pomegranate': 'pomegranate.jpg',
    'Lentil': 'lentil.jpg',
    'Blackgram': 'blackgram.jpg',
    'Mungbean': 'mungbean.jpg',
    'Mothbeans': 'mothbeans.jpg',
    'Pigeonpeas': 'pigeonpeas.jpg',
    'Kidneybeans': 'kidneybeans.jpg',
    'Chickpea': 'chickpea.jpg',
    'Coffee': 'coffee.jpg'
}

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        n=int(request.form['nitrogen'])
        p=int(request.form['phosphorus'])
        k=int(request.form['potassium'])
        t=float(request.form['temperature'])
        h=float(request.form['humidity'])
        ph=float(request.form['ph'])
        r=float(request.form['rainfall'])
        
        user_data=[[n,p,k,t,h,ph,r]]
        user_data=scaler.transform(user_data)
        prediction=kmeans.predict(user_data)

        for key,val in dt.items():
            if val==prediction :
                ls=key
        ls=ls.capitalize()
        
        # Get crop image, default to a placeholder if not found
        crop_image = CROP_IMAGES.get(ls, 'default.jpg')
        
        return render_template('result.html', prediction=ls, crop_image=crop_image)
    
if __name__ == '__main__':
    app.run(debug=True)
