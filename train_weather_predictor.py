from models import ImageDateRegressionModel, save_model, load_model
from weather_dataset import WeatherDataset, LocalWeatherDataset
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch


def train(args, device):

    model = ImageDateRegressionModel(layers=[32,64], normalize_input=True).to(device)

    current_time = datetime.now().strftime('%b-%d')
    print(current_time)
    max_epochs = 60
    scheduler_enabled = True


    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    #dataset = WeatherDataset(bucket_name='austin-gibs-images', transform=transform)
    dataset = LocalWeatherDataset("austin_weather_data/", transform=transform)
    # Define the sizes for your train, test, and validation sets
    train_size = int(0.7 * len(dataset))  # 70% for training
    test_size = int(0.15 * len(dataset))  # 15% for testing
    val_size = len(dataset) - train_size - test_size  # Remaining 15% for validation

    # Split the dataset
    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

    # Create DataLoaders for each set
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    criterion = nn.MSELoss()

    model.train()
    dataset_size = len(train_loader)
    print(dataset_size)
    print("Starting training loop")
    for epoch in range(max_epochs):
        print(f"Training epoch {str(epoch)}")
        i = 0
        total_train_loss = 0.0
        for batch_x, date_x, batch_y in train_loader:
            if device is not None:
                batch_x, date_x, batch_y = batch_x.to(device), date_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            logits = model(batch_x, date_x)
            loss = criterion(logits, batch_y)
            total_train_loss += loss
            if torch.isnan(loss.to('cpu')):
                print(batch_x)
                print(logits)
                print(batch_y)
                print("Loss is NaN")
            loss.backward()
            optimizer.step()
            i += 1

        train_loss = total_train_loss / len(train_loader)
        print(f"total_train_loss for epoch {epoch}: {train_loss}")

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x, date_x, batch_y in val_loader:
                if device is not None:
                    batch_x, date_x, batch_y = batch_x.to(device), date_x.to(device), batch_y.to(device)
                logits = model(batch_x, date_x)
                loss = criterion(logits, batch_y)
                total_val_loss += loss

        val_loss = total_val_loss / len(val_loader)

        print(f"total_val_loss for epoch {epoch}: {val_loss}")

        if scheduler_enabled:
            scheduler.step(val_loss)

    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for batch_x, date_x, batch_y in test_loader:
            if device is not None:
                batch_x, date_x, batch_y = batch_x.to(device), date_x.to(device), batch_y.to(device)
            logits = model(batch_x, date_x)
            loss = criterion(logits, batch_y)
            total_test_loss += loss
    test_loss = total_test_loss / len(test_loader)
    print(f"total_test_loss: {test_loss}")
    save_model(model, current_time)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir')
    # Put custom arguments here

    args = parser.parse_args()
    # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")

    # Set the device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    train(args, device)
