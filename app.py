import torchaudio
from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition
from speechbrain.utils.checkpoints import Checkpointer
import os
from flask import Flask, request, jsonify
import random
# from classifier import EncoderClassifier

# model = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="pretrained_models/lang-id-voxlingua107-ecapa")
model = EncoderClassifier.from_hparams(
    source='model/epaca/1988/save/CKPT+2024-02-15+14-26-50+00')

# print("model loaded")


app = Flask(__name__)


@app.route('/predict_lang', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['file']
        except KeyError:
            return jsonify({'error': 'No file found in request'})
        try:
            random_number = random.randint(0, 100000)
            file_name = f"output_{random_number}.wav"
            file.save(file_name)
            prediction = model.classify_file(file_name)
            os.remove(file_name)
            return jsonify({'language': prediction[3]})
        except Exception as e:
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
