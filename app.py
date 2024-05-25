from flask import Flask, request, jsonify
import gpt_2_simple as gpt2
import os

app = Flask(__name__)

# Define the model name
model_name = "124M"  # Change this to the model you want to use (e.g., 124M, 355M, 774M, 1558M)

# Start a TensorFlow session and load the model
session = gpt2.start_tf_sess()
gpt2.load_gpt2(session, model_name=model_name)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = gpt2.generate(session, model_name=model_name, prefix=user_input, return_as_list=True)[0]
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
