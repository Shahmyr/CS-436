import gpt_2_simple as gpt2

# Start a TensorFlow session
sess = gpt2.start_tf_sess()

# Download the 124M model (if not already downloaded)
gpt2.download_gpt2(model_name="124M")

# Load the model
gpt2.load_gpt2(sess, model_name="124M")

print("GPT-2 model loaded successfully!")
