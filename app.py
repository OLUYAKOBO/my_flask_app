"""import numpy as np
from flask import Flask, request, jsonify, render_template

#create flask app

app = Flask(__name__)

#load the pickle model
import pickle
model = pickle.load(open('model.pkl','rb'))

@app.route("/")

def Home():
    return render_template("index2.html")

@app.route("/predict", methods = ["POST"])

def predict():
    #int_features = [int(x) for x in request.form.values()]
    app_mode = int(request.form['Application mode'])
    app_order = int(request.form['Application order'])
    course = int(request.form['Course'])

    features = np.array([[app_mode,app_order,course]])#.reshape(1,-1)

    prediction = model.predict(features)
    status = ['Graduated','Dropout']

    #prediction = model.predict(features)
    #prediction_text = status[prediction[0]]

    #return render_template("index2.html", prediction = prediction)

    if prediction == 0:
        return render_template("index2.html", prediction_text = "The student Dropped out of school")
    else:
        return render_template("index2.html", prediction_text = "The student Graduated")

if __name__ == "__main__":
    app.run(debug = True)"""




import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the pickle model
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def Home():
    return render_template("index2.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Extract form data using correct field names
    app_mode = int(request.form['app_mode'])
    app_order = int(request.form['app_order'])
    course = int(request.form['course'])

     # Create input array for the model
    init_features = np.array([[app_mode, app_order, course]])

    #print("Initial features", init_features)

    features = scaler.transform(init_features)


    #print("Transformed features", features)
    # Make prediction
    prediction = model.predict(features)[0]  # Ensure you're extracting the first value

    # Return appropriate message
    if prediction == 0:
        prediction_text = "The student Dropped out of school"
    else:
        prediction_text = "The student Graduated"
    
    return render_template("index2.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)



