from model_training import models
import torch.onnx

model = models.load_model([32, 64], "Nov-01")
dummy_image = torch.rand((3, 512, 512)).unsqueeze(0)
dummy_date = torch.rand((1,4))
torch.onnx.export(model, args=(dummy_image, dummy_date), f="Model-Nov-01.onnx", input_names=['image_tensor', 'date_features'], export_params=True)