import torch
import torchvision
import boto3
import logging
import torch.nn as nn
logging.basicConfig(level=logging.INFO)
class ImageDateRegressionModel(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.block = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )

        def forward(self, x):
            return self.block(x)
    def __init__(self, layers=[], n_input_channels=3, normalize_input=False):

        super().__init__()
        L = [
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
        ]
        c = 32
        for ly in layers:
            L.append(self.Block(c, ly, stride=ly//c))
            c = ly
        self.normalize_input = normalize_input
        self.conv_layers = torch.nn.Sequential(*L)
        self.dropout = torch.nn.Dropout(0.5)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.img_linear = torch.nn.Linear(c, 6)
        self.relu = torch.nn.ReLU()
        self.date_linear = torch.nn.Linear(4, 2)
        self.classifier = torch.nn.Linear(8, 2)

    def forward(self, x, date_feature):
        if self.normalize_input:
            transform = torchvision.transforms.Normalize(mean=[0.4701, 0.4308, 0.3839], std=[0.2595, 0.2522, 0.2541])
            x = transform(x)
        x = self.conv_layers(x)
        x = self.gap(x)
        #x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.img_linear(x)) # [B, c] -> [B, 32]
        date_x = self.relu(self.date_linear(date_feature))
        x = torch.cat((x, date_x), dim=1)
        x = self.classifier(x)
        return x

class WeatherEncoder(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.block = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=n_input, out_channels=n_output, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1)),
                torch.nn.ReLU(),
                torch.nn.Conv3d(in_channels=n_output, out_channels=n_output, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1)),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d(kernel_size=(1, 2, 2),stride=(1,2,2), padding=(0, 1, 1))
            )

        def forward(self, x):
            return self.block(x)
    def __init__(self, layers=[], n_input_channels=3, n_embed=64, normalize_input=True):

        super().__init__()
        L = [
            torch.nn.Conv3d(in_channels=n_input_channels, out_channels=32, kernel_size=(1, 7, 7),
                            stride=(1, 2, 2), padding=(0, 3, 3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=(1, 3, 3),stride=(1,2,2), padding=(0, 1, 1))
        ]
        c = 32
        for ly in layers:
            L.append(self.Block(c, ly, stride=ly//c))
            c = ly
        self.normalize_input = normalize_input
        self.conv_layers = torch.nn.Sequential(*L)
        self.dropout = torch.nn.Dropout(0.5)
        self.gap = torch.nn.AdaptiveAvgPool3d((None, 1, 1))
        self.img_linear = torch.nn.Linear(c, n_embed-2)
        self.relu = torch.nn.ReLU()
        self.date_linear = torch.nn.Linear(4, 2)

    def forward(self, x, date_feature):
        if self.normalize_input:
            transform = torchvision.transforms.Normalize(mean=[0.4701, 0.4308, 0.3839], std=[0.2595, 0.2522, 0.2541])
            x = transform(x)
        x = x.transpose(1, 2)
        #print(f"before conv_layers: {x.shape}")
        x = self.conv_layers(x)
        #print(f"after_conv: {x.shape}")
        x = self.gap(x)
        #print(f"after_gap: {x.shape}")
        x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        #print(f"after_reshape: {x.shape}")
        x = self.relu(self.img_linear(x)) # [B, c] -> [B, 32]
        #print(f"after_img_linear: {x.shape}")
        date_x = self.relu(self.date_linear(date_feature))
        #print(f"after_date: {date_x.shape}")
        x = torch.cat((x, date_x), dim=2)
        #print(f"after_concat: {x.shape}")
        return x

class Head(nn.Module):
    def __init__(self, seq_length, d_model, num_heads, d_internal):
        super().__init__()
        self.K = nn.Linear(d_model, d_internal)
        self.Q = nn.Linear(d_model, d_internal)
        self.V = nn.Linear(d_model, d_internal)
        self.w0 = nn.Linear(d_internal, d_model // num_heads)
        self.register_buffer('tril', torch.tril(torch.ones(seq_length, seq_length)))

    def forward(self, input_vecs):
        keys = self.K(input_vecs)  # B, L, d_internal
        d_k = keys.shape[-1]
        queries = self.Q(input_vecs)  # B, L, d_internal
        value = self.V(input_vecs)  # B, L, d_internal
        weights = torch.matmul(queries, keys.transpose(-2, -1)) * d_k ** -0.5  # L, L
        weights = weights.masked_fill(self.tril == 0, float('-inf'))
        attention = torch.softmax(weights, dim=-1)

        logit = torch.matmul(attention, value)  # B, L, d_internal
        logit = self.w0(logit)
        return logit

class FeedFoward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)
class MultiHeadAttention(nn.Module):

    def __init__(self, seq_length, d_model, num_heads, d_internal):
        super().__init__()
        self.heads = nn.ModuleList([Head(seq_length, d_model, num_heads, d_internal) for _ in range(num_heads)])
        self.linear1 = nn.Linear(d_model, d_model)

    def forward(self, input_vecs):
        out = torch.cat([head(input_vecs) for head in self.heads], dim=-1)
        out = self.linear1(out)
        return out

class MHATransformerLayer(nn.Module):
    def __init__(self, seq_length, d_model, num_heads, d_internal):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(seq_length, d_model, num_heads, d_internal)
        self.ffwd = FeedFoward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, input_vecs):
        x = self.multi_head_attention(self.ln1(input_vecs))
        x += input_vecs
        x = x + self.ffwd(self.ln2(x))

        return x
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20):
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)


    def forward(self, x, batched=False):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)
class MHATransformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_layers, num_heads):
        super().__init__()
        self.num_positions = num_positions
        self.L = []
        for ly in range(num_layers):
            self.L.append(MHATransformerLayer(num_positions, d_model, num_heads, d_internal))
        self.transformer_layers = nn.Sequential(*self.L)
        self.classifier = nn.Linear(d_model, vocab_size)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, num_positions=num_positions)
        self.layer_norm = nn.LayerNorm(d_model)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, indices, batched=False):
        logit = self.embedding(indices)
        logit = self.pos_embedding(logit, batched=batched)
        logit = self.transformer_layers(logit)
        logit = self.classifier(logit)
        logit = self.softmax(logit)
        if batched:
            return logit
        else:
            return logit.squeeze(0)


class WeatherSequenceModel(torch.nn.Module):
    def __init__(self,  seq_len, n_embed, n_attn, num_layers, num_heads, layers=[], n_input_channels=3,):

        super().__init__()
        self.weather_encoder = WeatherEncoder(layers=layers, n_input_channels=n_input_channels)
        self.L = []
        for ly in range(num_layers):
            self.L.append(MHATransformerLayer(seq_len, n_embed, num_heads, n_attn))
        self.pos_embedding = PositionalEncoding(n_embed, num_positions=seq_len)
        self.transformer_layers = nn.Sequential(*self.L)
        self.classifier = nn.Linear(n_embed, 2)

    def forward(self, x, date_feature, batched=False):
        logit = self.weather_encoder(x, date_feature) # B, seq_len, n_embed
        #print(f"after_encoder: {logit.shape}")
        #logit = self.pos_embedding(logit, batched=batched)
        logit = self.transformer_layers(logit) # B, seq_len, n_embed
        #print(f"after_transformers: {logit.shape}")
        logit = self.classifier(logit) # B, seq_len, 2 # switch to CLS method
        #print(f"after_classifier: {logit.shape}")
        if batched:
            return logit
        else:
            return logit.squeeze(0)

def save_model(model, id, device, bucket_name='austin-weather-prediction-models'):
    if isinstance(model, ImageDateRegressionModel):
        dummy_image = torch.rand((3, 512, 512)).unsqueeze(0).to(device)
        dummy_date = torch.rand((1, 4)).to(device)
        model_filename = f"Model-{id}.onnx"
        torch.onnx.export(model, args=(dummy_image, dummy_date), f=model_filename,
                          input_names=['image_tensor', 'date_features'], export_params=True)
        s3 = boto3.client('s3')
        try:
            s3.upload_file(model_filename, bucket_name, model_filename)
            logging.info(f"Model '{model_filename}' uploaded to S3 bucket '{bucket_name}' successfully.")
            return True
        except Exception as e:
            logging.error(f"Error uploading to s3: {e}")
    raise ValueError("model type '%s' not supported!" % str(type(model)))

def save_sequential_model(model, id, device, bucket_name='austin-weather-prediction-models'):
    if isinstance(model, WeatherSequenceModel):
        dummy_image = torch.rand((20, 3, 512, 512)).unsqueeze(0).to(device)
        dummy_date = torch.rand((1, 20, 4)).to(device)
        model_filename = f"Model-Sequential-{id}.onnx"
        torch.onnx.export(model, args=(dummy_image, dummy_date), f=model_filename,
                          input_names=['image_tensor', 'date_features'], export_params=True)
        s3 = boto3.client('s3')
        try:
            s3.upload_file(model_filename, bucket_name, model_filename)
            logging.info(f"Model '{model_filename}' uploaded to S3 bucket '{bucket_name}' successfully.")
            return True
        except Exception as e:
            logging.error(f"Error uploading to s3: {e}")
    raise ValueError("model type '%s' not supported!" % str(type(model)))

def load_model(layers, id):
    from torch import load
    from os import path
    r = ImageDateRegressionModel(layers=layers)
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), f'Model_{id}.th'), map_location='cpu'))
    return r
