import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("GEMINI_API_KEY")

# Configure the AI model
genai.configure(api_key=api_key)

# Flask app setup
app = Flask(__name__)

# AI Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the generative model
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction="To create a system that predicts the cost of building a house in India.",
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        # Get the list of inputs from the frontend
        user_inputs = request.json.get('inputs', [])
        # Combine inputs into a formatted string
        user_input_text = "\n".join(user_inputs)

        # Start chat with the AI (initialize conversation)
        chat_session = model.start_chat(history=[])
        # Send the user's input to the AI and get the response
        response = chat_session.send_message(user_input_text)

        # Return AI response
        return jsonify({'response': response.text})

    except Exception as e:
        # Handle errors and send response back
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
