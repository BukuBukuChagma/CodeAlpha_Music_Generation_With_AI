# AI Music Generation Web Application

This web application generates music using Meta's MusicGen Small model with 8-bit quantization for optimized performance on GPUs. The application provides a user-friendly interface for creating unique musical compositions based on text prompts and musical parameters.

![Web Page Screenshot](/screenshot.png)

## Features
image.png
- Generate music based on text descriptions/prompts
- Adjust musical parameters (genre, key, tempo, complexity)
- Direct audio output in WAV format
- Built-in audio player for immediate playback
- Responsive UI design for desktop and mobile devices
- Automatic fallback to smaller models if the primary model fails to load
- Detailed error messages with troubleshooting guidance

## Technical Implementation

### Backend
- Flask web server for serving the application
- MusicGen Small model with 8-bit quantization for efficient memory usage
- Fallback mechanism to handle network issues during model download
- Optimized for running on consumer-grade GPUs with limited memory
- Proper MIME type handling for audio playback and downloads
- Local model caching to prevent repeated downloads

### Frontend
- Clean and responsive design with modern CSS
- Interactive controls for music parameters with real-time feedback
- Enhanced error handling with persistent error messages for user guidance
- Real-time audio playback with streaming capabilities
- Separate download functionality for saving generated audio

## Project Structure

```
.
├── app.py                 # Main Flask application
├── models.py              # MusicGen model implementation
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
├── README.md              # This documentation
├── static/                # Static files
│   ├── css/               # CSS stylesheets
│   │   └── styles.css     # Main stylesheet
│   └── audio/             # Generated audio files
├── templates/             # HTML templates
│   ├── index.html         # Main application page
│   ├── 404.html           # 404 error page
│   └── 500.html           # 500 error page
└── model_cache/           # Cached model files
```

## Requirements

The application requires the following Python packages:

```
flask           # Web framework
torch           # PyTorch for deep learning
transformers    # Hugging Face Transformers for model loading
scipy           # For audio processing
numpy           # For numerical operations
bitsandbytes    # For 8-bit quantization
accelerate      # For optimized model loading
```

## Setup and Usage

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/CodeAlpha_Music_Generation_With_AI.git
   cd CodeAlpha_Music_Generation_With_AI
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Using the application:
   - Enter a descriptive prompt for your desired music
   - Select a musical genre and key
   - Adjust the tempo slider (60-180 BPM)
   - Set the complexity level (1-10, affects duration)
   - Click "Generate Music" to create your composition
   - Once generated, use the built-in player to listen
   - Download the WAV file if desired

## System Requirements

- Python 3.8 or higher
- 4GB+ GPU memory for optimal performance (NVIDIA GPU recommended)
- 8GB+ system RAM
- Modern web browser with audio support
- Internet connection for initial model download

## Known Issues and Troubleshooting

### Network Issues

- **Model Download Failures**: The MusicGen model (approximately 2GB) is downloaded from Hugging Face when first used
  - **Symptoms**: 403 Forbidden errors, timeouts, or "Network error" messages
  - **Solutions**: 
    - Ensure you have a stable internet connection
    - Try again at a less busy time
    - Consider using a VPN if your network restricts large downloads
    - Download the model separately and place it in the `model_cache` directory

- **Slow First Start**: The first generation takes longer as the model is loaded into memory
  - **Expected behavior**: First generation may take 1-10 minutes, subsequent generations are faster

### Memory Issues

- **Out of Memory Errors**: The model requires significant GPU memory
  - **Symptoms**: CUDA out of memory errors, application crashes
  - **Solutions**:
    - Lower the complexity setting to reduce memory usage
    - Close other GPU-intensive applications
    - The application will attempt to use a smaller model if available
    - Set `CUDA_VISIBLE_DEVICES=-1` to force CPU usage (much slower)

### Audio Playback Issues

- **Browser Compatibility**: Some browsers may handle WAV files differently
  - **Symptoms**: Audio doesn't play, or downloads automatically
  - **Solutions**:
    - Try a different browser (Chrome/Firefox recommended)
    - Check that your browser supports audio playback
    - Make sure browser audio isn't muted

- **Audio Quality Issues**: Generated audio may have artifacts or noise
  - **Solutions**:
    - Try different prompts
    - Experiment with different genre/key combinations
    - Use more specific musical descriptions

### Alternative Models
If you continue to experience downloading issues, consider:
- Using a local copy of the model by downloading it separately and specifying a local path 

## Contributing

Contributions to improve the AI Music Generation Web Application are welcome! Here's how you can contribute:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and implement tests if applicable
4. **Commit your changes**:
   ```bash
   git commit -am 'Add some feature'
   ```
5. **Push to the branch**:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Create a new Pull Request**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Meta's MusicGen](https://huggingface.co/facebook/musicgen-small) for the underlying music generation model
- [Hugging Face](https://huggingface.co/) for hosting and providing access to AI models
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [PyTorch](https://pytorch.org/) for the machine learning backend


