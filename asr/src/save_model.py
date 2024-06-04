from transformers import AutoModelForSpeechSeq2Seq, AutoModelForCausalLM, AutoProcessor, pipeline
import torch
model_id = "distil-whisper/distil-large-v3"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa")
model.save_pretrained("./fastest-whisper")
