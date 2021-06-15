import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
import flask
import pickle
import os
import random
import knn_images
import vqa_main
import guessing_bot.bot_utils as ut
from guessing_bot.G_Bot import G_Bot
import torch as T

dataDir		=   'static/VQA_dataset'
versionType =   'v2_' # this should be '' when using VQA v2.0 dataset
taskType    =   'OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    =   'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType =   'val2014'
annFile     =   f'{dataDir}/Annotations/{versionType}{dataType}_{dataSubType}_annotations.json'
quesFile    =   f'{dataDir}/Questions/{versionType}{taskType}_{dataType}_{dataSubType}_questions.json'
imgDir 		=   f'{dataDir}/Images/{dataType}/{dataSubType}/'

with open('pickles/image_encodings.p', 'rb') as f:
    image_encoding = pickle.load(f)
    #print(image_encoding)

params = {
    'vocabSize': 20573,
    'embedSize': 300,
    'rnnHiddenSize': 512,
    'dialogInputSize': 512,
    'numLayers': 2,
    'imgFeatureSize': 2048,
    'numRounds': 3,
    'numEpochs': 1
}

app = Flask(__name__)
guessing_bot = G_Bot(params)
#guessing_bot.load_state_dict(T.load(''))
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('guessWhich.html')


@app.route('/start_game',methods=['POST'])
def start():
    image_idx = random.randint(0,len(image_encoding.keys()))
    image_idx = list(image_encoding.keys())[image_idx]
    params = [str(x) for x in request.form.values()]
    all_idx, chosen = knn_images.get_nearest_images_idx(image_idx, image_encoding, int(params[0]))

    print("CHOSEN (SERVER):", chosen)
    print("all idxs", all_idx)
    data = {}
    data['Indexes'] = [ut.imgID_2_filename(dataSubType, idx) for idx in all_idx]
    data['Chosen'] = ut.imgID_2_filename(dataSubType, chosen)

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
    pic_idx = params[0]
    question = params[1]

    data = {}

    print('ask: ', pic_idx)
    idx = ut.filename_2_idx(pic_idx)
    print('ask_2: ', idx)
    encoding = image_encoding[str(idx)]

    df = vqa_main.model_work(None, question, encoding)
    confidence = df['confidence'][0]
    answer = df['answers'][0]
    print(confidence)
    if confidence < 50:
        if confidence < 20:
            data["Success"] = False
            data['Answer'] = "Couldn't understand the question. Please ask another one :D"
        else:
            data["Success"] = True
            data['Answer'] = "I'm not completely sure, but I think " + str(answer)
    else:
        data['Answer'] = df['answers'][0]

    """
    y = [int(idx) for idx in image_encoding.keys()]
    q_idx, a_idx = ut.convert_statements_to_idx("how many people?", "2", guessing_bot.token_to_ix)
    q, a = T.tensor(np.array([q_idx])), T.tensor(np.array([a_idx]))
    ql, al = T.tensor(np.array([[len(q_idx)]])), T.tensor(np.array([[len(a_idx)]]))
    guessing_bot.observe(ques=q, anws=a, ql=ql, al=al)
    idx = guessing_bot(y=y)
    real_idx = y[idx]
    filename = ut.imgID_2_filename(dataSubType, real_idx)
    data["Bot_guess"] = filename
    """
    data["Bot_guess"] = "COCO_val2014_000000000164.jpg"


    return flask.jsonify(data)


@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')



if __name__ == "__main__":
    app.run(debug=True, host=os.getenv('IP', '0.0.0.0'), port=int(os.getenv('PORT', 4444)))

