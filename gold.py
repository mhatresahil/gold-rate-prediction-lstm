import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt

#Load the data
df = pd.read_csv('gold_rate.csv')

#Preprocess the data
def preprocess_gold(df):
    """
    Preprocesses gold price data for time series prediction.
    
    Args:
        df (pandas.DataFrame): Input dataframe containing 'Date' and 'Value' columns
        
    Returns:
        tuple: (train_dataloader, test_dataloader) containing batched training and test data
        
    The preprocessing steps include:
    1. Sorting data chronologically
    2. Creating lagged features (t-1 to t-5) for time series prediction
    3. Normalizing features using StandardScaler
    4. Splitting into train (75%) and test (25%) sets
    5. Converting to PyTorch tensors and creating DataLoaders
    """
    # Sort chronologically by date
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    
    # Create 5 lagged features (t-1 to t-5) for time series prediction
    lags = 5
    for i in range(1,lags+1):
        df[f'gold_val_lag{i}'] = df['Value'].shift(i)  
    df = df[lags+1:] # Remove rows with NaN from shifting
    
    # Normalize all features using StandardScaler
    norm_cols = [f'gold_val_lag{i}' for i in  range(1,lags+1)]
    norm_cols.append('Value')
    scaler = StandardScaler()
    df.loc[:,norm_cols] = scaler.fit_transform(df[norm_cols])
    
    # Split into training (75%) and testing (25%) sets
    train_data = df.sample(frac = 0.75, random_state = 19)
    test_data = df.drop(train_data.index)
    
    # Convert to PyTorch tensors
    train_features = torch.tensor(train_data[[f'gold_val_lag{i}' for i in range(1, lags + 1)]].values, dtype=torch.float32)
    train_target = torch.tensor(train_data['Value'].values, dtype=torch.float32)
    
    test_features = torch.tensor(test_data[[f'gold_val_lag{i}' for i in range(1, lags + 1)]].values, dtype=torch.float32)
    test_target = torch.tensor(test_data['Value'].values, dtype=torch.float32)
    
    # Create DataLoaders with batching
    batch_size = 32
    
    train_dataset = TensorDataset(train_features, train_target)
    test_dataset = TensorDataset(test_features, test_target)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    return train_dataloader, test_dataloader

train_dataloader, test_dataloader = preprocess_gold(df)

#Define a Baseline Neural Network model
#class goldrate_predictorNN(nn.Module):
#    def __init__(self):
#        super(goldrate_predictorNN, self).__init__()
#        self.fn1 = nn.Linear(5,10)
#        self.fn2 = nn.Linear(10,20)
#        self.fn3 = nn.Linear(20,20)
#        self.fn4 = nn.Linear(20,10)
#        self.fn5 = nn.Linear(10,5)
#        self.fn6 = nn.Linear(5,1)
#        self.relu = nn.Tanh()
#        
#    def forward(self, x):
#        x = self.fn1(x)
#        x = self.relu(x)
#        x = self.fn2(x)
#        x = self.relu(x)
#        x = self.fn3(x)
#        x = self.relu(x)
#        x = self.fn4(x)
#        x = self.relu(x)
#        x = self.fn5(x)
#        x = self.relu(x)
#        x = self.fn6(x)
#        return x

#Define a LSTM model
class goldrate_predictorlstm(nn.Module):
    def __init__(self):
        super(goldrate_predictorlstm, self).__init__()
        # LSTM layer with input size 5 (number of features), hidden size 5, single layer
        self.lstm1 = nn.LSTM(input_size=5, hidden_size=5, num_layers=1, batch_first=True)
        # Final linear layer to produce single output prediction
        self.fn3 = nn.Linear(5, 1)
        
    def forward(self, x):
        # Add a time dimension to input tensor (batch_size, 1, features)
        x = x.unsqueeze(1)
        # Pass through LSTM layer, ignoring hidden state
        out, _ = self.lstm1(x)
        # Remove time dimension since we only have one timestep
        out = out.squeeze(1)
        # Final linear layer to get prediction
        out = self.fn3(out)
        return out

#model = goldrate_predictorNN()
model = goldrate_predictorlstm()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

num_epochs = 100
train_loss = []
for epoch in range(num_epochs):
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))
        loss.backward()
        optimizer.step()
    train_loss.append(loss.item())
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    test_loss = 0.0
    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        test_loss += criterion(outputs, targets.view(-1, 1)).item()

    avg_test_loss = test_loss / len(test_dataloader)
    print(f'Average Test Loss: {avg_test_loss:.4f}')
    
epochs = (range(1, num_epochs+1))
plt.plot(epochs, train_loss,'b',label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()