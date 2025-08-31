import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)
print("Image size:", image.height, image.width)  # [480, 640]

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
patch_size = model.config.patch_size
print("Patch size:", patch_size) # 16
print("Num register tokens:", model.config.num_register_tokens) # 4

inputs = processor(images=image, return_tensors="pt")
print("Preprocessed image size:", inputs.pixel_values.shape)  # [1, 3, 224, 224]

batch_size, _, img_height, img_width = inputs.pixel_values.shape
num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
num_patches_flat = num_patches_height * num_patches_width

with torch.inference_mode():
  outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)  # [1, 1 + 4 + 256, 384]
assert last_hidden_states.shape == (batch_size, 1 + model.config.num_register_tokens + num_patches_flat, model.config.hidden_size)

cls_token = last_hidden_states[:, 0, :]
patch_features_flat = last_hidden_states[:, 1 + model.config.num_register_tokens:, :]
patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))