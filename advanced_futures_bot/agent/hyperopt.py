from typing import Any, Dict, Optional
import optuna
from ray import tune

class HyperOpt:
    """
    PPO + LSTM + Transformer agent için hyperparameter tuning.
    Hem Optuna hem Ray Tune desteği.
    """
    def __init__(self, trainer: Any, method: str = "optuna", config: Optional[Dict[str, Any]] = None):
        self.trainer = trainer
        self.method = method
        self.config = config or {}

    def _objective_optuna(self, trial: optuna.Trial) -> float:
        # Örnek: PPO parametreleri
        lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        entropy = trial.suggest_uniform("entropy_coef", 0.0, 0.05)
        clip_range = trial.suggest_uniform("clip_range", 0.1, 0.3)
        # Trainer'a parametreleri ilet
        result = self.trainer.train_with_params({
            "learning_rate": lr,
            "batch_size": batch_size,
            "entropy_coef": entropy,
            "clip_range": clip_range
        })
        return result["reward"]  # maximize

    def optimize_optuna(self, n_trials: int = 20):
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective_optuna, n_trials=n_trials)
        print("Best params:", study.best_params)
        return study.best_params

    def optimize_ray(self, num_samples: int = 20):
        search_space = {
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([32, 64, 128, 256]),
            "entropy_coef": tune.uniform(0.0, 0.05),
            "clip_range": tune.uniform(0.1, 0.3)
        }
        def trainable(config):
            result = self.trainer.train_with_params(config)
            tune.report(reward=result["reward"])
        analysis = tune.run(trainable, config=search_space, num_samples=num_samples)
        print("Best config:", analysis.get_best_config(metric="reward", mode="max"))
        return analysis.get_best_config(metric="reward", mode="max")

    def optimize(self, n_trials: int = 20):
        if self.method == "optuna":
            return self.optimize_optuna(n_trials)
        elif self.method == "ray":
            return self.optimize_ray(n_trials)
        else:
            raise ValueError("method optuna veya ray olmalı") 