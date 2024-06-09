import torch
from src.mlops_project.entity.config_entity import ModelInferenceConfig
from src.mlops_project.utils.model_architecture import NeuralNetwork

class ModelInference:
    def __init__(self, config: ModelInferenceConfig) -> None:
        self.config=config
    
    def predict(self, input):

        model = NeuralNetwork(self.config.input_dim, 
                              self.config.hidden1_dim, 
                              self.config.hidden2_dim, 
                              self.config.output_dim)
        
        model.load_state_dict(torch.load("model_path.pth"))

        input_tensor = torch.tensor(input, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_class = torch.max(output, dim=1)

        return predicted_class
        