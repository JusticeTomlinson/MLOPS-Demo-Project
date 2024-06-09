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
from src.mlops_project.utils.model_architecture import NeuralNetwork






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

        model = NeuralNetwork(self.config.input_dim, self.config.hidden1_dim, self.config.hidden2_dim, self.config.output_dim).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        results = self.train_model(model, train_loader, criterion, optimizer, self.config.num_epochs, os.path.join(self.config.root_dir, self.config.model_path))
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

    