from typing import Any, Dict, Optional
import optuna
from ray import tune
import logging
import os

logger = logging.getLogger("HyperOpt")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class HyperOpt:
    """
    Production-level hyperparameter tuning.
    - Optuna & Ray Tune entegrasyonu
    - Checkpoint, early stopping, logging, exception handling
    """
    def __init__(self, trainer: Any, method: str = "optuna", config: Optional[Dict[str, Any]] = None):
        self.trainer = trainer
        self.method = method
        self.config = config or {}
        self.checkpoint_dir = self.config.get("checkpoint_dir", "hyperopt_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.early_stopping = self.config.get("early_stopping", True)
        self.patience = self.config.get("patience", 5)

    def _objective_optuna(self, trial: optuna.Trial) -> float:
        try:
            lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
            entropy = trial.suggest_uniform("entropy_coef", 0.0, 0.05)
            clip_range = trial.suggest_uniform("clip_range", 0.1, 0.3)
            result = self.trainer.train_with_params({
                "learning_rate": lr,
                "batch_size": batch_size,
                "entropy_coef": entropy,
                "clip_range": clip_range
            })
            reward = result["reward"]
            trial.set_user_attr("reward", reward)
            # Checkpoint
            if hasattr(self.trainer, "save"):
                self.trainer.save(os.path.join(self.checkpoint_dir, f"trial_{trial.number}.pt"))
            logger.info(f"Optuna trial {trial.number}: reward={reward}")
            return reward
        except Exception as e:
            logger.error(f"Optuna trial {trial.number} failed: {e}")
            return -1e9

    def optimize_optuna(self, n_trials: int = 20):
        study = optuna.create_study(direction="maximize")
        if self.early_stopping:
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=self.patience)
            study.pruner = pruner
        study.optimize(self._objective_optuna, n_trials=n_trials)
        logger.info(f"Optuna best params: {study.best_params}")
        return study.best_params

    def optimize_ray(self, num_samples: int = 20):
        search_space = {
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([32, 64, 128, 256]),
            "entropy_coef": tune.uniform(0.0, 0.05),
            "clip_range": tune.uniform(0.1, 0.3)
        }
        def trainable(config):
            try:
                result = self.trainer.train_with_params(config)
                tune.report(reward=result["reward"])
                # Checkpoint
                if hasattr(self.trainer, "save"):
                    self.trainer.save(os.path.join(self.checkpoint_dir, f"ray_{tune.get_trial_id()}.pt"))
            except Exception as e:
                logger.error(f"Ray Tune trial failed: {e}")
                tune.report(reward=-1e9)
        analysis = tune.run(trainable, config=search_space, num_samples=num_samples)
        best_config = analysis.get_best_config(metric="reward", mode="max")
        logger.info(f"Ray Tune best config: {best_config}")
        return best_config

    def optimize(self, n_trials: int = 20):
        if self.method == "optuna":
            return self.optimize_optuna(n_trials)
        elif self.method == "ray":
            return self.optimize_ray(n_trials)
        else:
            raise ValueError("method optuna veya ray olmalı") 