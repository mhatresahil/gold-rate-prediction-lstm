import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

#Load the data
df = pd.read_csv('gold_rate.csv')

#Preprocess the data
def preprocess_gold(df):
    #Sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    
    #Creating Lag Features
    lags = 5
    for i in range(1,lags+1):
        df[f'gold_val_lag{i}'] = df['Value'].shift(i)  
    df = df[lags+1:]
    
    #Normalizing Data
    norm_cols = [f'gold_val_lag{i}' for i in  range(1,lags+1)]
    norm_cols.append('Value')
    scaler = StandardScaler()
    df[norm_cols] = scaler.fit_transform(df[norm_cols])
    
    #Splitting into training and testing data
    train_data = df.sample(frac = 0.75, random_state = 19)
    test_data = df.drop(train_data.index)
    
    train_features = torch.tensor(train_data[[f'gold_val_lag{i}' for i in range(1, lags + 1)]].values, dtype=torch.float32)
    train_target = torch.tensor(train_data['Value'].values, dtype=torch.float32)
    
    test_features = torch.tensor(test_data[[f'gold_val_lag{i}' for i in range(1, lags + 1)]].values, dtype=torch.float32)
    test_target = torch.tensor(test_data['Value'].values, dtype=torch.float32)
    
    batch_size = 32
    
    train_dataset = TensorDataset(train_features, train_target)
    test_dataset = TensorDataset(test_features, test_target)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    return train_dataloader, test_dataloader

train_dataloader, test_dataloader = preprocess_gold(df)