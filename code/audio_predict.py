
from utils.audio_model import get_cnn_model
from utils.wav2mfcc import wav2mfcc
from keras.models import load_model
import h5py
import glob
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'


def get_test_data(audio_dir, scene):
    spec = []
    for audio_path in glob.glob("%s/*" % (os.path.join(audio_dir, scene))):
        spec.append(wav2mfcc(audio_path))
    return np.asarray(spec)


if __name__ == "__main__":
    sound_dir = "./data/audio"
    scene = "nnew1"
    env_dir = "./data/environment"

    x_test = get_test_data(sound_dir, scene)
    print(x_test.shape)
    model  = load_model('./data/pretrained_model/audio_preditor.h5')
    # model = get_cnn_model((128, 128, 2))
    pred = model.predict(x_test, batch_size=10)
    print(pred.shape)

    h5_file = h5py.File("%s/%s.h5" % (env_dir, scene), "r+")
    if "predict_source" in h5_file.keys():
        h5_file.__delitem__("predict_source")
    h5_file.create_dataset("predict_source", data=pred)