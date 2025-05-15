# HuggingFace Transformers on Azure Functions

## Getting Started

### Prerequisites

1. Install [Python 3.12+](https://www.python.org/downloads/)

    - (Recommended) Create a virtual environment using the tool or your choice, e.g. venv or conda.

1. To test locally create a file named `local.settings.json` in this directory with this structure:

    ```json
    {
      "IsEncrypted": false,
      "Values": {
        "AzureWebJobsStorage": "",
        "FUNCTIONS_WORKER_RUNTIME": "python",
        "MODEL_NAME": "<HuggingFace model name>",
        "TASK_TYPE": "<inference task type, same as with >",
        "HF_HOME": "<path for model cache>"
      }
    }
    ```

### Installation

(Recommended) Activate your virtual environment

User `pip install -r .\requirements.txt` to install all dependencies

## Usage

### Azure Function

To run the Azure function use `func start`.
