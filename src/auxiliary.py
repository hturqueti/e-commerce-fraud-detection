# Libraries
import yaml

from pathlib import Path

# Constants
project_path = Path('.').resolve().parent

def load_parameters() -> dict[str, str]:
    with open(project_path.joinpath("parameters.yaml"), 'r') as file:
        parameters = yaml.safe_load(file)
    return parameters