Repo containing code to train and evaluate Aecoc categorization model for Algori with new data.

## Installation

1. Clone the repository
2. Install required system dependencies:
  - Python 3.11
  - Azure CLI (or anything else that can authenticate to Azure services)
  - ODBC Driver for SQL Server (to download data from Azure SQL); tested with version 17.9, newer versions may work
  - Latest nvidia and cuda drivers for GPU acceleration; tested with CUDA 12.5, cudnn 9.1 and nvidia 550.90.07
3. Install the required packages using the following command:
```bash
pip install .
```

## Usage

### Getting training data

1. Copy .env.example to .env and fill in the required values.
2. Run `python scripts/download_raw_data.py` to download the raw data from Azure SQL. Later scripts will use same defaults as this one, but feel free to modify them as needed. It will create a `data/raw` directory with the following files:
  - `fuzzy_mappings.csv`: mappings of various ocr mistakes to correct values
  - `verified_ocrs.csv`: categorizied data with verified OCRs
3. Run `python scripts/prepare_training_data.py` to prepare the training data.

Instead of using python to download data, you can just check the `data/queries` directory and download data manually.

### Training the model

Run `python scripts/train_model.py` to train the model. It will create a `models` directory where it will store model checkpoints.
You can always see training progress in your console. You can also check training history by going to [Azure ML Studio/Jobs](https://ml.azure.com/experiments?wsid=/subscriptions/4b0f0735-0433-402c-88cd-9f2c162d22fa/resourcegroups/promos/providers/Microsoft.MachineLearningServices/workspaces/promos_ml_models&tid=3f2bd556-eb58-466d-aa6c-2c5da9934f10) and checking the latest run.

### Evaluating the model

I used jupyter notebooks to evaluate the model. `notebooks/evaluate.ipynb` to be exact

## To be done

- Teach model about trash data
- Evaluate model by retailer
- Train model on all available data (i.e., on validation and test slices after final evaluation)
- Include scraped data in training; we can probably mapped scraped categories to internal categories using Kmeans or something similar