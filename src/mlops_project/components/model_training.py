import os
from src.mlops_project import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, TensorDataset
import torch
from src.mlops_project.config.configuration import ModelTrainingConfig



def resample_classes(dataframe):
    df_majority = dataframe[dataframe["CHDRisk"] == 0]
    df_minority = dataframe[dataframe["CHDRisk"] == 1]

    minority_count = len(df_minority)

    df_majority_downsampled = df_majority.sample(n=minority_count, random_state=42)  # Ensuring reproducibility

    df_balanced = pd.concat([df_minority, df_majority_downsampled])

    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    #Turn all items within the table to float
    df_balanced = df_balanced.astype(float)
    return df_balanced




class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        super(NeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden1_dim)
        self.layer2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output_layer = nn.Linear(hidden2_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = torch.sigmoid(self.output_layer(x))
        return x
    


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig) -> None:
        self.config = config
    
    def train(self):

        train_data = pd.read_csv(self.config.train_data_path)
        train_data = train_data.dropna()

        resampled_train_data = resample_classes(train_data)

        device = torch.device("cpu")


        features = resampled_train_data.drop('CHDRisk', axis=1).values
        targets = resampled_train_data['CHDRisk'].values
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)

        model = NeuralNetwork(self.config.input_dim, self.config.hidden1_dim, self.config.hidden2_dim, self.config.output_dim).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        results = self.train_model(model, train_loader, criterion, optimizer, self.config.num_epochs, os.path.join(self.config.root_dir, self.config.model_name))
        return results

    
    def train_model(self, model, train_loader, criterion, optimizer, num_epochs, model_path):
        """ Train the model """
        results = {"losses": [], "accuracies": []}

        for epoch in range(num_epochs):
            total = 0
            correct = 0

            for inputs, labels in train_loader:

                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            results["losses"].append(loss)
            results["accuracies"].append(accuracy)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
            torch.save(model.state_dict(), model_path)
        return results

    