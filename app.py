import numpy as np
from flask import Flask, request, jsonify, render_template
import flask
import pickle
import os
import random
import knn_images

app = Flask(__name__)
app._static_folder = os.path.abspath("templates/static/")
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():

	
	return start()
	#return flask.jsonify(data)  
	#return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route('/start_game',methods=['POST'])
def start():
	image_idx = random.randint(0,1000)
	print(image_idx)
	all_idx = knn_images.get_nearest_images_idx(image_idx) #TODO no ficheiro tem que se fazer um metodo que receba um idx e que retorne um array de idx
	print("all idxs",all_idx)
	data = {}
	data['Indexes'] = [str(x) for x in all_idx]
	data['Chosen'] = str(image_idx) + ".jpg"

	return flask.jsonify(data)  


@app.route('/ask',methods=['POST'])
def ask():
    '''
    For rendering results on HTML GUI
    '''
    params = [str(x) for x in request.form.values()]
    #data = flask.request.files this is a dict, and also this is for files
    #it depends whether you do it as a form as before, or send it as a file
    #will be dependent of the HTML

    #assumindo que ja tenho o id e a question que ia no form
    print(params)
    pic_name = params[0]
    question = params[1]

    data = {}

    df = vqa_main.model_work(pic_name, question) #TODO neste tem de se fazer um metodo que receba o pic_name e a question e retorne as respostas e confianças (o dataframe)
    if df['confidence'][0] < 0.50:
    	data["Success"] = True
    	data['Answer'] = "Couldn't understand the question. Please ask another one :D"
    else:
    	data['Answer'] = df['answers'][0]

    
    return flask.jsonify(data)






if __name__ == "__main__":
    app.run(debug=True)
