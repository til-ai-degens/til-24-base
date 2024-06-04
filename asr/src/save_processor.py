from transformers import AutoModelForSpeechSeq2Seq, AutoModelForCausalLM, AutoProcessor, pipeline
import torch
model_id = "distil-whisper/distil-large-v3"

processor = AutoProcessor.from_pretrained(model_id)
processor.save_pretrained("./fastest-whisper-processor")
