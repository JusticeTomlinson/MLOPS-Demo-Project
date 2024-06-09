
from src.mlops_project.config.configuration import ConfigurationManager
from src.mlops_project.components.model_inference import ModelInference


class ModelInferencePipeline:
    def __init__(self) -> None:
        pass
    def predict(self, input):
        
        config = ConfigurationManager()
        model_inference_config = config.get_model_inference_config()
        model_inference_config = ModelInference(config=model_inference_config)
        prediction = model_inference_config.predict(input)

        return prediction
        