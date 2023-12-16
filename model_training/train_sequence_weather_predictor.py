from models import WeatherSequenceModel, save_model
from weather_dataset import LocalWeatherSequenceDataset, download_dataset_from_s3
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torch
import logging
logging.basicConfig(level=logging.INFO)

def train(args, device):
    try:
        if args.download:
            logging.info("Downloading data from s3...")
            download_dataset_from_s3()
            logging.info("Downloading data finished.")
        # seq_len, n_embed, n_attn, num_layers, num_heads, layers=[], n_input_channels=3,
        seq_len = 20
        model = WeatherSequenceModel(seq_len, 64, 64, 1, 4,  layers=[32,64]).to(device)

        current_time = datetime.now().strftime("%b-%Y")
        logging.info(current_time)
        max_epochs = 30
        scheduler_enabled = True


        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        #dataset = WeatherDataset(bucket_name='austin-gibs-images', transform=transform)
        dataset = LocalWeatherSequenceDataset("./austin_weather_data/", seq_len, transform=transform)
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
        logging.info(dataset_size)
        logging.info("Starting training loop")
        for epoch in range(max_epochs):
            logging.info(f"Training epoch {str(epoch)}")
            i = 0
            total_train_loss = 0.0
            for batch_x, date_x, batch_y in train_loader:
                if device is not None:
                    batch_x, date_x, batch_y = batch_x.to(device), date_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                print(f"{batch_x.shape} {date_x.shape} {batch_y.shape}")
                logits = model(batch_x, date_x)
                loss = criterion(logits, batch_y)
                total_train_loss += loss
                if torch.isnan(loss.to('cpu')):
                    logging.info(batch_x)
                    logging.info(logits)
                    logging.info(batch_y)
                    logging.info("Loss is NaN")
                loss.backward()
                optimizer.step()
                i += 1

            train_loss = total_train_loss / len(train_loader)
            logging.info(f"total_train_loss for epoch {epoch}: {train_loss}")

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

            logging.info(f"total_val_loss for epoch {epoch}: {val_loss}")

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
        logging.info(f"total_test_loss: {test_loss}")
        save_model(model, current_time, device)
    except Exception as e:
        logging.error(f"Training failed with exception {e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # Put custom arguments here
    parser.add_argument('--download', action='store_true', help='If set, download data')
    args = parser.parse_args()

    # Set the device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    logging.info(f"Using device: {device}")

    train(args, device)
