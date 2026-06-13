import numpy as np
import onnxruntime as ort

session = ort.InferenceSession('/model/model.onnx')

def get_predict(data):
    features = np.array(data['features'], dtype=np.float32)
    res = session.run(None, {'features':features})
    return res