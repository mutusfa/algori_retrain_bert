import argparse
from pathlib import Path
import sys

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import mlflow
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning import Trainer

from retrain_bert.data import MixedDatasetModule
from retrain_bert.data.loaders import load_labels, load_train_data
from retrain_bert.model import AlgoriAecocCategorization
from retrain_bert.preprocessor import get_labels_conf


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, default="retrain-aecoc-categorization")
    parser.add_argument("--artifact-path", type=Path, default=Path("models")
    return parser.parse_args(argv)


def main(args):
    args.artifact_path.mkdir(parents=True, exist_ok=True)

    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        mlf_logger = MLFlowLogger(
            experiment_name=args.experiment_name,
            tracking_uri=mlflow_tracking_uri,
            synchronous=False,
            log_model="all",
            run_id=mlflow.active_run().info.run_id,
        )
        train_data = load_train_data()
        labels = load_labels()
        labels_conf = get_labels_conf(labels)

        data_module = MixedDatasetModule(
            train_data=train_data["train"],
            val_data=train_data["validation"],
            test_data=train_data["unseen_categories"],
            labels_conf=labels_conf,
            batch_size=128,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.artifact_path,
            filename="{epoch}-{val_accuracy_level_3:.3f}",
            save_top_k=-1,  # save all models
            verbose=True,
            monitor="val_accuracy_level_3",
            mode="max",  # save models with minimum validation loss
        )

        model = AlgoriAecocCategorization(labels_conf=labels_conf)
        trainer = Trainer(
            max_epochs=5,
            logger=mlf_logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(
            model,
            datamodule=data_module,
        )
        model.fine_tune()
        trainer.fit_loop.max_epochs = trainer.current_epoch + 5
        trainer.fit(model, datamodule=data_module)

        mlflow.pytorch.log_model(
            model,
            "aecoc-classifier",
            registered_model_name="CustomModel",
        )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    sys.exit(main(args))
