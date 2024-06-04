docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-TEAM-NAME/TEAM-NAME-asr:finalsimport torch
import io
import soundfile as sf
import noisereduce as nr
from transformers import AutoModelForSpeechSeq2Seq, AutoModelForCausalLM, AutoProcessor, pipeline

class ASRManager:
    def __init__(self):
        # Define the model checkpoint and device type
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "fastest-whisper" # put in our model here!
        print(f"Loading model...")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa"
        )
        self.model.to(self.device)
        
        
        self.processor = AutoProcessor.from_pretrained("fastest-whisper-processor")
        
        # Initialize the speech recognition pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=self.device,
        )

    def transcribe(self, audio_bytes: bytes) -> str:
        with io.BytesIO(audio_bytes) as audio_file:
            audio_array, sample_rate = sf.read(audio_file)
        
        
        noise_clip = audio_array[:int(0.5 * sample_rate)]  # first 0.5 seconds 
        audio_clean = nr.reduce_noise(y=audio_array, sr=sample_rate, noise_clip=noise_clip)

        
        # Process the audio and generate transcription using the pipeline
        #transcription = self.pipe(audio_array)
        transcription = self.pipe(audio_clean)
        
        return transcription['text']




# import torch
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# import base64
# import io
# import soundfile as sf
# import noisereduce as nr
# import numpy as np
# # USE THIS IF ABOVE DOESN'T WORK

# class ASRManager:
#     def __init__(self):
#         # Load the Whisper model and processor
#         self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language='en')
#         print("Downloading the poopy model...")
#         self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
#         self.model.config.forced_decoder_ids = None

#         # Load model to device (CPU or GPU)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#         self.model.eval()  # Set model to evaluation mode

#     def transcribe(self, audio_bytes: bytes) -> str:
#         with io.BytesIO(audio_bytes) as audio_file:
#             audio_array, sample_rate = sf.read(audio_file)

#         # Apply noise reduction with adjusted parameters
#         # audio_array = nr.reduce_noise(y=audio_array, sr=sample_rate)

#         # Process the audio array
#         inputs = self.processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")

#         # Move inputs to device
#         input_features = inputs["input_features"].to(self.device)

#         # Generate transcription
#         with torch.no_grad():
#             predicted_ids = self.model.generate(input_features)

#         # Decode the transcription
#         transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

#         return transcription[0]
