import hydra
from omegaconf import DictConfig

from train_gpt2 import GPT2TextClassification


@hydra.main(version_base=None, config_path="configs/predict", config_name="config.yaml")
def main(config: DictConfig) -> None:
    print(config)
    assert config.text
    assert config.model_path
    model_path = config.model_path
    text = config.text
    model = GPT2TextClassification.load_from_checkpoint(model_path)

    inputs = model.tokenizer([text], return_tensors='pt')["input_ids"]
    labels = model(inputs)
    print(labels)


if __name__ == "__main__":
    main()
