import torch
import torchvision

class ImageDateRegressionModel(torch.nn.Module):
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




def save_model(model, id):
    from torch import save
    from os import path
    if isinstance(model, ImageDateRegressionModel):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), f'model_{id}.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))

def load_model(layers, id):
    from torch import load
    from os import path
    r = ImageDateRegressionModel(layers=layers)
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), f'model_{id}.th'), map_location='cpu'))
    return r
