import os
import torch
import numpy as np
import time
from scipy.io import wavfile

class MusicGenModel:
    def __init__(self, audio_folder):
        self.model = None
        self.processor = None
        self.initialized = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_folder = audio_folder
        self.initialization_error = None
        self.model_name = "facebook/musicgen-small"  # Using small model by default
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
        
        # Create model cache directory if it doesn't exist
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        self.load_model()
    
    def load_model(self):
        try:
            # Import inside function to avoid loading on server start
            from transformers import AutoProcessor, MusicgenForConditionalGeneration, BitsAndBytesConfig
            import torch.quantization
            
            print(f"Loading MusicGen model {self.model_name} on {self.device}...")
            print(f"Available CUDA devices: {torch.cuda.device_count()}")
            if torch.cuda.is_available():
                print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
                print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            
            # Load processor and model with 8-bit quantization
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.model_path
            )
            
            # Configure 8-bit quantization settings
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            
            # For 8-bit models, DO NOT use .to(device) as it's not supported
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                cache_dir=self.model_path
            )
            
            # Set model to evaluation mode
            self.model.eval()
            self.initialized = True
            print(f"MusicGen model {self.model_name} loaded successfully")
            
            if torch.cuda.is_available():
                print(f"CUDA memory after loading: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                
        except Exception as e:
            error_str = str(e)
            print(f"Error loading MusicGen model {self.model_name}: {error_str}")
            
            # Store error message based on the type of error
            if "403 Forbidden" in error_str or "Read timed out" in error_str or "AccessDenied" in error_str:
                self.initialization_error = "Network error while downloading model. There seems to be connection issues or permission problems accessing the model files from Hugging Face. Please check your internet connection and try again later."
            elif "CUDA out of memory" in error_str:
                self.initialization_error = "GPU memory error. Your GPU doesn't have enough memory to load the model. Try closing other applications using the GPU."
            elif ".to is not supported for `8-bit`" in error_str:
                self.initialization_error = "The model is already loaded on the correct device. Please try again."
            else:
                self.initialization_error = f"Failed to initialize the model: {error_str}"
                
            self.initialized = False
    
    def generate(self, prompt, duration=5.0, genre="classical", key="C", tempo=120):
        """Generate music with MusicGen model based on text prompt and return the audio file path"""
        if not self.initialized:
            if self.initialization_error:
                raise RuntimeError(self.initialization_error)
            else:
                raise RuntimeError("Model failed to initialize. Please check system requirements and try again.")
        
        # Create a descriptive prompt including musical parameters
        full_prompt = f"Generate {genre} music in the key of {key} at {tempo} BPM. {prompt}"
        print(f"Generating with prompt: {full_prompt}")
        
        try:
            # Set generation parameters - optimized for GPU
            inputs = self.processor(
                text=[full_prompt],
                padding=True,
                return_tensors="pt",
            )
            
            # For 8-bit models, inputs should be moved to the same device as the model
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Calculate max_new_tokens based on duration (44.1kHz sampling rate)
            # MusicGen uses ~1500 tokens per 10 seconds of audio
            max_new_tokens = min(int(duration * 150), 1000)  # Limit to 1000 tokens for memory
            
            torch.cuda.empty_cache()  # Clear cache before generation
            
            # Generate audio tokens
            with torch.cuda.amp.autocast():  # Use mixed precision
                with torch.no_grad():  # Disable gradient calculation
                    generation_output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                    )
            
            # Convert to audio and ensure proper data type for WAV file
            audio_values = generation_output.cpu().numpy()[0, 0]
            
            # Convert to float32 (required by scipy's wavfile)
            audio_values = audio_values.astype(np.float32)
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio_values)) > 0:
                audio_values = audio_values / np.max(np.abs(audio_values))
            
            # Convert to WAV format
            sample_rate = 32000  # MusicGen's output sample rate
            timestamp = int(time.time())
            
            # Create WAV filename with parameters
            wav_filename = f"{genre}_{key}_{tempo}_{timestamp}.wav"
            wav_path = os.path.join(self.audio_folder, wav_filename)
            
            # Save the audio file - handle with try/except for proper error messages
            try:
                wavfile.write(wav_path, sample_rate, audio_values)
                print(f"Successfully saved WAV file to {wav_path}")
            except Exception as wav_error:
                print(f"Error saving WAV file: {str(wav_error)}")
                raise RuntimeError(f"Failed to save audio file: {str(wav_error)}")
            
            return wav_filename
            
        except RuntimeError as e:
            # Handle out of memory errors with more specific message
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                raise RuntimeError("GPU out of memory. Try reducing complexity, closing other GPU-intensive applications, or using a machine with more memory.")
            else:
                raise RuntimeError(f"Failed to generate music: {str(e)}")
        except Exception as e:
            torch.cuda.empty_cache()  # Always clean up in case of error
            raise RuntimeError(f"An unexpected error occurred during music generation: {str(e)}") 