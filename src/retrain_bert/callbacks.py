import mlflow
import tensorflow as tf


class MLFlowCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            mlflow.log_metric("train_loss_epoch", logs.get("loss"), step=epoch)
            mlflow.log_metric("val_loss_epoch", logs.get("val_loss"), step=epoch)
            mlflow.log_metric(
                "unseen_categories_loss_epoch",
                logs.get("unseen_categories_loss"),
                step=epoch,
            )

        # Save model weights
        model_path = f"model_epoch_{epoch}.h5"
        self.model.save_weights(model_path)
        mlflow.log_artifact(model_path)


class ValidationMetrics(tf.keras.callbacks.Callback):
    def __init__(self, validation_sets):
        super().__init__()
        self.validation_sets = validation_sets

    def on_epoch_end(self, epoch, logs=None):
        for name, dataset in self.validation_sets.items():
            results = self.model.evaluate(dataset, verbose=0)
            for metric, value in zip(self.model.metrics_names, results):
                mlflow.log_metric(f"{name}_{metric}_epoch", value, step=epoch)
