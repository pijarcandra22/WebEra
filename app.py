from flask import Flask, render_template, request, url_for, redirect,session
from werkzeug.utils import secure_filename
import pickle
from sklearn import neighbors
from scipy.io import wavfile
import librosa
import librosa.display
import numpy as np
import uuid
import os

app = Flask(__name__)
app = Flask(__name__,template_folder='temp')

model = pickle.load(open("py/model.sav", 'rb'))

ALLOWED_EXTENSIONS = {'wav', 'mp3'}

def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_audio(directory,name_file):
  if name_file not in request.files:
    return False
  file = request.files[name_file]
  if file.filename == '':
    return False
  if file and allowed_file(file.filename):
    app.config['UPLOAD_FOLDER'] = directory
    filename = secure_filename(file.filename)
    formatfile=filename.split('.')
    newfilename=str(uuid.uuid4().hex)+'.'+formatfile[1]
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    os.rename(os.path.join(app.config['UPLOAD_FOLDER'], filename),os.path.join(app.config['UPLOAD_FOLDER'], newfilename))
    return newfilename

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process():
  audio=upload_audio("data","audio")
  if not audio:
    return "None"
  
  feature = get_features("data/"+audio)
  X = []
  for ele in feature:
    X.append(ele)
  
  X = np.array(X)
  predict = model.predict(X)
  print(predict)
  return str(predict.mean())

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.70):
    return librosa.effects.time_stretch(data, rate)
    
def higher_speed(data, speed_factor = 1.25):
    return librosa.effects.time_stretch(data, speed_factor)

def lower_speed(data, speed_factor = 0.75):
    return librosa.effects.time_stretch(data,speed_factor)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def extract_features(data):
    result = np.array([])

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=22050,n_mfcc=200).T, axis=0)
    tempo, beats = librosa.beat.beat_track(y=data, sr=22050)
    result = np.hstack((result, mfcc, np.mean(tempo), np.mean(beats)))

    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=3, offset=0.5, res_type='kaiser_fast') 
    
    #without augmentation
    res1 = extract_features(data)
    result = np.array(res1)
    
    #noised
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2)) # stacking vertically
    
    #stretched
    stretch_data = stretch(data)
    res3 = extract_features(stretch_data)
    result = np.vstack((result, res3))
    
    #shifted
    shift_data = shift(data)
    res4 = extract_features(shift_data)
    result = np.vstack((result, res4))
    
    #pitched
    pitch_data = pitch(data, sample_rate)
    res5 = extract_features(pitch_data)
    result = np.vstack((result, res5)) 
    
    #speed up
    higher_speed_data = higher_speed(data)
    res6 = extract_features(higher_speed_data)
    result = np.vstack((result, res6))
    
    #speed down
    lower_speed_data = higher_speed(data)
    res7 = extract_features(lower_speed_data)
    result = np.vstack((result, res7))
    
    return result

if __name__=='__main__':
  app.run()