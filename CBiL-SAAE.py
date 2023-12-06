import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from collections import defaultdict

df_glucose = pd.read_excel('Reference_BG.xlsx', skiprows=1, header=None)
directory_path = './'

fs = 1000
fifteen_seconds_data_points = 15 * fs
five_minutes_data_points = 5 * 60 * fs
segments = []
mapped_glucose_values = []

mat_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.mat')])
excel_days = df_glucose.iloc[0].astype(int).astype(str).values
segments_per_day = []
for mat_file in tqdm(mat_files, desc="Processing MAT files"):
    day_name = mat_file.split('.')[0]
    day_segments = 0  

    if day_name in excel_days:
        col_index = df_glucose.columns[excel_days == day_name].to_list()[0]
        glucose_values = df_glucose[col_index].iloc[1:].dropna().values

        data = loadmat(os.path.join(directory_path, mat_file))
        df = pd.DataFrame(data.get('PPG', []), columns=['PPG'])

        for i, glucose_value in enumerate(glucose_values):
            start_idx = i * five_minutes_data_points
            end_idx = (i + 1) * five_minutes_data_points
            five_minute_chunk = df['PPG'].iloc[start_idx:end_idx].values

            for j in range(0, five_minutes_data_points, fifteen_seconds_data_points):
                segment = five_minute_chunk[j:j+fifteen_seconds_data_points]
                segments.append(segment)
                mapped_glucose_values.append(glucose_value)
                day_segments += 1
                
    else:
        print(f"No matching glucose data found for day {day_name}. Skipping...")
    
    segments_per_day.append(day_segments)

def clarke_error_grid_regions_percentage(reference, prediction):
    total_points = len(reference)
    regions_count = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

    for ref, pred in zip(reference, prediction):
        if (ref <= 70 and 0.5 * ref <= pred <= 1.5 * ref) or \
           (70 < ref <= 290 and 0.7 * ref <= pred <= 1.3 * ref) or \
           (ref > 290 and pred <= ref + 110):
            regions_count['A'] += 1
        elif (ref < 70 and ref + 30 <= pred <= 1.5 * ref) or \
             (70 <= ref <= 290 and (0.7 * ref <= pred < 0.8 * ref or 1.2 * ref <= pred <= 1.28 * ref)) or \
             (ref > 290 and ref + 110 < pred <= 1.5 * ref):
            regions_count['B'] += 1
        elif (70 <= ref < 180 and pred > 1.5 * ref) or (ref >= 180 and 1.5 * ref < pred <= ref + 110):
            regions_count['C'] += 1
        elif (ref >= 70 and pred < 0.7 * ref) or (ref <= 290 and pred < ref - 110):
            regions_count['D'] += 1
        else:
            regions_count['E'] += 1

    regions_percentage = {region: (count / total_points) * 100 for region, count in regions_count.items()}
    return regions_percentage
    
segments = [seg for seg in segments if len(seg) == fifteen_seconds_data_points]
X = np.array(segments).reshape(-1, 1, fifteen_seconds_data_points)
y = np.array(mapped_glucose_values)[:len(segments)]

train_days = 100
train_indices = sum(segments_per_day[:train_days])

X_train = torch.tensor(X[:train_indices], dtype=torch.float32)
y_train = torch.tensor(y[:train_indices], dtype=torch.float32)
X_test = torch.tensor(X[train_indices:], dtype=torch.float32)  
y_test = torch.tensor(y[train_indices:], dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

device = torch.device("cuda:1")

class DeepSpatialAttention(nn.Module):
    def __init__(self, input_dim):
        super(DeepSpatialAttention, self).__init__()
       
        self.query = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim)
        )
        self.key = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim)
        )
        
    def forward(self, x):
        
        q = self.query(x) 
        k = self.key(x)   
        
        attn_weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)
        attn_output = torch.bmm(attn_weights, x)
        
        return attn_output, attn_weights

class ComplexLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(ComplexLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=0.5, 
                            bidirectional=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        x, _ = self.lstm(x)
        return x

class ComplexLSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ComplexLSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[ :, -1, :]) 
        return x

class EnhancedGlucoseModel(nn.Module):
    def __init__(self):
        super(EnhancedGlucoseModel, self).__init__()

    
        self.encoder_lstm1 = ComplexLSTMEncoder(15000, 256, 16)
        self.encoder_lstm2 = ComplexLSTMEncoder(512, 128, 8)
        self.encoder_lstm3 = ComplexLSTMEncoder(256, 64, 2)
   
        self.deep_spatial_attention = DeepSpatialAttention(128)
   
        self.decoder_lstm = ComplexLSTMDecoder(128, 64, 1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.encoder_lstm1(x)
        x = self.encoder_lstm2(x)
        x, attn_weights = self.deep_spatial_attention(x)
        x = self.encoder_lstm3(x)
        x = self.decoder_lstm(x)
        x = self.fc(x)
        return x

model = EnhancedGlucoseModel().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-3)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.92)

num_epochs = 50
best_val_loss = float('inf')
early_stop_patience = 15
no_improve_epochs = 0

for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    model.train()
    train_loss = 0  
    for batch_x, batch_y in tqdm(train_loader, desc="Training Batches", leave=False):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)  
        optimizer.zero_grad()        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y.unsqueeze(1))        
        loss.backward()
        optimizer.step()
        scheduler.step()        
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)
      
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
        
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), 'best_model_csa.pth')
    else:
        no_improve_epochs += 1
        if no_improve_epochs == early_stop_patience:
            print("Early stopping...")
            model.load_state_dict(torch.load('best_model_csa.pth'))
            break

model.eval()
y_pred = [] 
with torch.no_grad():
    for batch_x, _ in tqdm(val_loader, desc="Predicting"):
        batch_x = batch_x.to(device)  
        outputs = model(batch_x)
        y_pred.extend(outputs.cpu().squeeze().tolist()) 
        

y_test_np = y_test.numpy()
y_pred_np = np.array(y_pred)

rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_np))
mard = np.mean(np.abs((y_test_np - y_pred_np) / y_test_np)) * 100

print(f"RMSE: {rmse:.2f}")
print(f"MARD: {mard:.2f}%")

regions_percentage = clarke_error_grid_regions_percentage(y_test_np, y_pred_np)
print(f"Clarke Error Grid Analysis:")
for region, percentage in regions_percentage.items():
    print(f"Region {region}: {percentage:.2f}%")
    
#10-folds
# Reshape and filter
segments = [seg for seg in segments if len(seg) == fifteen_seconds_data_points]
X = np.array(segments).reshape(-1, 1, fifteen_seconds_data_points)
y = np.array(mapped_glucose_values)[:len(segments)]

# KFold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Variables to store aggregated metrics across folds
total_rmse = 0
total_mard = 0
total_ceg_regions_percentage = defaultdict(float)

device = torch.device("cuda:2")

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Processing Fold {fold + 1}")
    
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]

    # Convert to PyTorch tensors
    X_train_fold = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
    y_train_fold = torch.tensor(y_train_fold, dtype=torch.float32).to(device)
    X_val_fold = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
    y_val_fold = torch.tensor(y_val_fold, dtype=torch.float32).to(device)

    # Create DataLoaders for this fold
    train_dataset_fold = TensorDataset(X_train_fold, y_train_fold)
    val_dataset_fold = TensorDataset(X_val_fold, y_val_fold)
    
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=32, shuffle=True)
    val_loader_fold = DataLoader(val_dataset_fold, batch_size=32)


    class DeepSpatialAttention(nn.Module):
        def __init__(self, input_dim):
            super(DeepSpatialAttention, self).__init__()

            # Adding more transformations for the query and key generation
            self.query = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, input_dim // 4),
                nn.ReLU(),
                nn.Linear(input_dim // 4, input_dim)
            )
            self.key = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, input_dim // 4),
                nn.ReLU(),
                nn.Linear(input_dim // 4, input_dim)
            )

        def forward(self, x):
            batch_size, seq_len, _ = x.size()

            x_reshaped = x.contiguous().view(-1, x.size(-1))

            q = self.query(x_reshaped).view(batch_size, seq_len, -1)
            k = self.key(x_reshaped).view(batch_size, seq_len, -1)

            attn_weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)
            attn_output = torch.bmm(attn_weights, x)

            return attn_output, attn_weights

    class ComplexLSTMEncoder(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(ComplexLSTMEncoder, self).__init__()

            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                                num_layers=num_layers, batch_first=True, dropout=0.5, 
                                bidirectional=True)
            self.hidden_size = hidden_size

        def forward(self, x):
            x, _ = self.lstm(x)
            return x

    class ComplexLSTMDecoder(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(ComplexLSTMDecoder, self).__init__()

            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                                num_layers=1, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x, _ = self.lstm(x)
            x = x[:, -1, :]  # Selecting the last timestep
            x = self.fc(x)
            return x

    class EnhancedGlucoseModel(nn.Module):
        def __init__(self):
            super(EnhancedGlucoseModel, self).__init__()

            self.encoder_lstm1 = ComplexLSTMEncoder(15000, 256, 16)
            self.encoder_lstm2 = ComplexLSTMEncoder(512, 128, 8)
            self.encoder_lstm3 = ComplexLSTMEncoder(256, 64, 2)

            self.deep_spatial_attention = DeepSpatialAttention(128)

            self.decoder_lstm = ComplexLSTMDecoder(128, 64, 1)
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            x = self.encoder_lstm1(x)
            x = self.encoder_lstm2(x)
            x, attn_weights = self.deep_spatial_attention(x)
            x = self.encoder_lstm3(x)
            x = self.decoder_lstm(x)
            return x

    # Instantiate the model for this fold
    model = EnhancedGlucoseModel().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.92)

    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    early_stop_patience = 15
    no_improve_epochs = 0

    # Training loop for this fold
    for epoch in tqdm(range(num_epochs), desc=f"Training Epochs Fold {fold + 1}"):
        model.train()
        train_loss = 0  
        for batch_x, batch_y in tqdm(train_loader_fold, desc="Training Batches", leave=False):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  
            optimizer.zero_grad()        
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))        
            loss.backward()
            optimizer.step()
            scheduler.step()        
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader_fold)
    
        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader_fold:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                val_loss += loss.item()
    
        avg_val_loss = val_loss / len(val_loader_fold)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'best_model_csa.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs == early_stop_patience:
                print("Early stopping...")
                model.load_state_dict(torch.load('best_model_csa.pth'))
                break

    # Prediction for this fold
    y_pred_fold = []
    with torch.no_grad():
        for batch_x, _ in tqdm(val_loader_fold, desc=f"Predicting Fold {fold + 1}"):
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            y_pred_fold.extend(outputs.cpu().squeeze().tolist())

    # Metrics for this fold
    y_val_fold_np = y_val_fold.cpu().numpy()
    y_pred_fold_np = np.array(y_pred_fold)
    
    rmse_fold = np.sqrt(mean_squared_error(y_val_fold_np, y_pred_fold_np))
    mard_fold = np.mean(np.abs((y_val_fold_np - y_pred_fold_np) / y_val_fold_np)) * 100

    print(f"Fold {fold + 1} RMSE: {rmse_fold:.2f}")
    print(f"Fold {fold + 1} MARD: {mard_fold:.2f}%")
    
    regions_percentage_fold = clarke_error_grid_regions_percentage(y_val_fold_np, y_pred_fold_np)
    for region, percentage in regions_percentage_fold.items():
        print(f"Region {region}: {percentage:.2f}%")
        total_ceg_regions_percentage[region] += percentage

    total_rmse += rmse_fold
    total_mard += mard_fold

# Averaging metrics across all folds
print(f"Average RMSE across 10-folds: {total_rmse / 10:.2f}")
print(f"Average MARD across 10-folds: {total_mard / 10:.2f}%")

print(f"Clarke Error Grid Analysis across 10-folds:")
for region, percentage in total_ceg_regions_percentage.items():
    print(f"Region {region}: {percentage / 10:.2f}%")