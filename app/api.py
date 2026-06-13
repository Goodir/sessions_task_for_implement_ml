from flask import Flask, request, jsonify
import model_handler

app = Flask(__name__)

@app.route('/')
def index():
    return 'Test message. The server is running'

@app.route('/health', methods=['GET'])
def health():
      return jsonify({'status':'healthy'}), 200

@app.route('/predict', methods=['POST'])
def pred():
	data = request.get_json()

	result = model_handler.get_predict(data)
	print(result[0])
	return jsonify({"prediction": result[0].tolist(), 'probability':result[1]})