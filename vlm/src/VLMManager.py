from typing import List
import torch
from transformers import pipeline
import io
from PIL import Image

class VLMManager:
    def __init__(self):
        torch.cuda.empty_cache()
        # load device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        #self.checkpoint = "google/owlvit-base-patch32"
        local_checkpoint = "./VLM_model"
        
        # detector
        self.detector = pipeline(model=local_checkpoint, task="zero-shot-object-detection", device=0 if torch.cuda.is_available() else -1)
        print("detector loaded!")
    
    def process_image(self, image_bytes: bytes):
        # Create a BytesIO object from the bytes
        image_stream = io.BytesIO(image_bytes)
        # Use Image.open to open the image from the BytesIO stream
        image = Image.open(image_stream)
        return image
    
    def convert_to_coco(self, box):
        if not box:
            return [0,0,0,0]
        # original box [x_min, y_min, x_max, y_max]
        # convert to [left, top, width, height]
        left = box["xmin"]
        top = box["ymin"]
        width = box["xmax"] - left
        height = box["ymax"] - top

        return [left, top, width, height]

    def identify(self, image: bytes, caption: str) -> List[int]:
        p_image = self.process_image(image)
        prediction = self.detector(
            p_image,
            caption,
            threshold=0.0000001,
            top_k=1
        )
        
        # get box
        print(prediction)
        if not prediction:
            return [0,0,0,0]
        box = prediction[0]["box"]
            
        
        # process box into coco format
        coco_box = self.convert_to_coco(box)
            
        return coco_box
