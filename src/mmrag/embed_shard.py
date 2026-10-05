import torch, numpy as np, webdataset as wds
from transformers import CLIPModel, CLIPProcessor

model = processor = device = None

def load():
    global model, processor, device

    if model is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # uses mps for faster compute
        device = "mps" if torch.backends.mps.is_available() else "cpu" 

        # tells model to use mps and eval not train
        model = model.to(device).eval()

    return model, processor, device


def embed_shard(file_path, batch_size=256):
    model, processor, device = load()
    
    # reads tar, decodes to PIL image, retrieve jpg, txt, and key files only, and process batch_size images simultaneously
    ds = wds.WebDataset(file_path).decode("pil").to_tuple('jpg', 'txt', '__key__').batched(batch_size)

    vectors, captions, keys = [], [], []

    # tells PyTorch to not retain computation information, since there's no training
    with torch.no_grad():
        for sample in ds:
            images_batched = sample[0]
            caps_batched = sample[1]
            keys_batched = sample[2]

            # preprocess and get features
            images_input = processor(images=images_batched, return_tensors="pt").to(device)
            embedded_image = model.get_image_features(**images_input)

            # if not tensor, convert to tensor
            if not isinstance(embedded_image, torch.Tensor):   
                embedded_image = embedded_image.pooler_output

            # L2 normalization
            embedded_image = embedded_image / embedded_image.norm(dim=-1, keepdim=True)

            # move off gpu (in order to save) and convert to numpy array
            vectors.append(embedded_image.cpu().numpy())

            captions += caps_batched
            keys += keys_batched

    # merge vectors into one grid and then turn into list of vectors (each row represents a vector)
    vectors = list(np.vstack(vectors).astype(np.float32))

    return vectors, captions, keys
