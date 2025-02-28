import os
import torch
import numpy as np
import time
import io
from flask import Flask, render_template, request, jsonify, send_file
from models import MusicGenModel

# Create output directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('static/audio', exist_ok=True)

app = Flask(__name__)
app.config['AUDIO_FOLDER'] = os.path.join(app.root_path, 'static', 'audio')

# Initialize model
music_model = None

def load_model():
    """Load the music generation model"""
    global music_model
    music_model = MusicGenModel(app.config['AUDIO_FOLDER'])

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_music():
    """Generate music based on parameters"""
    # Extract parameters from request
    tempo = int(request.form.get('tempo', 120))
    genre = request.form.get('genre', 'classical')
    key = request.form.get('key', 'C')
    complexity = int(request.form.get('complexity', 5))
    prompt = request.form.get('prompt', '')
    
    # Create a default prompt if none provided
    if not prompt:
        prompt = f"Create a {genre} piece with {complexity} complexity."
    
    # Map complexity to duration (1-10 to 5-20 seconds)
    duration = 5 + (complexity - 1) * 1.5
    
    try:
        if music_model is None:
            return jsonify({
                'status': 'error',
                'message': "Model not loaded. Please check server logs."
            })
        
        # Generate music using MusicGen model
        audio_filename = music_model.generate(
            prompt=prompt,
            duration=duration,
            genre=genre,
            key=key,
            tempo=tempo
        )
        
        return jsonify({
            'status': 'success',
            'message': f"Music generated successfully using MusicGen model!",
            'audio_file': f"/play_audio/{audio_filename}"
        })
        
    except RuntimeError as e:
        # Handle specific runtime errors with more detail
        print(f"Runtime error generating music: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })
    except Exception as e:
        # Handle any other unexpected errors
        print(f"Unexpected error generating music: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Failed to generate music: {str(e)}"
        })

@app.route('/play_audio/<filename>')
def play_audio(filename):
    """Serve the audio file for playback in the browser"""
    try:
        # Set mimetype to audio/wav for browser playback
        # Add additional headers to prevent download and improve caching
        response = send_file(
            os.path.join(app.config['AUDIO_FOLDER'], filename),
            mimetype='audio/wav',
            as_attachment=False,
            download_name=filename
        )
        # Add headers to prevent download behavior
        response.headers['Content-Disposition'] = 'inline'
        response.headers['Accept-Ranges'] = 'bytes'  # Enable range requests properly
        return response
    except Exception as e:
        print(f"Error serving audio file: {str(e)}")
        return "Audio file not found or could not be played", 404

@app.route('/download/<filename>')
def download_file(filename):
    """Download the generated audio file"""
    try:
        # Explicitly use attachment disposition for downloads
        response = send_file(
            os.path.join(app.config['AUDIO_FOLDER'], filename),
            mimetype='audio/wav',
            as_attachment=True,
            download_name=filename
        )
        # Force the Content-Disposition header to attachment
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
    except Exception as e:
        print(f"Error downloading audio file: {str(e)}")
        return "Audio file not found or could not be downloaded", 404

if __name__ == '__main__':
    # Load model in a separate thread to avoid blocking app startup
    import threading
    threading.Thread(target=load_model).start()
    app.run(debug=True) 