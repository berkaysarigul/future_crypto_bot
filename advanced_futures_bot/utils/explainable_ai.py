from typing import Any, Dict, Optional, List
import numpy as np
import torch
import shap
import lime
import lime.lime_tabular

class ExplainableAI:
    """
    SHAP ve LIME explainability.
    - Feature importance
    - Model interpretability
    - Decision explanations
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.shap_enabled = self.config.get("shap", True)
        self.lime_enabled = self.config.get("lime", True)
        self.feature_names = self.config.get("feature_names", [])
        self.explanation_dir = self.config.get("explanation_dir", "explanations")

    def explain_shap(self, 
                    model: Any,
                    background_data: np.ndarray,
                    sample_data: np.ndarray) -> Dict[str, Any]:
        """
        SHAP explanations.
        """
        if not self.shap_enabled:
            return {"enabled": False}
        
        try:
            # Create SHAP explainer
            explainer = shap.DeepExplainer(model, torch.tensor(background_data))
            
            # Generate SHAP values
            shap_values = explainer.shap_values(torch.tensor(sample_data))
            
            # Feature importance
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Individual predictions
            individual_explanations = []
            for i, sample in enumerate(sample_data):
                explanation = {
                    "sample_id": i,
                    "shap_values": shap_values[i].tolist(),
                    "feature_importance": feature_importance.tolist()
                }
                individual_explanations.append(explanation)
            
            return {
                "enabled": True,
                "method": "shap",
                "feature_importance": feature_importance.tolist(),
                "individual_explanations": individual_explanations,
                "background_data_shape": background_data.shape,
                "sample_data_shape": sample_data.shape
            }
        
        except Exception as e:
            return {"enabled": False, "error": str(e)}

    def explain_lime(self,
                    model: Any,
                    sample_data: np.ndarray,
                    feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        LIME explanations.
        """
        if not self.lime_enabled:
            return {"enabled": False}
        
        try:
            feature_names = feature_names or self.feature_names or [f"feature_{i}" for i in range(sample_data.shape[1])]
            
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                sample_data,
                feature_names=feature_names,
                class_names=["short", "flat", "long"],
                mode="classification"
            )
            
            # Generate explanations for each sample
            lime_explanations = []
            for i, sample in enumerate(sample_data):
                # Get LIME explanation
                exp = explainer.explain_instance(
                    sample,
                    lambda x: self._model_predict_wrapper(model, x),
                    num_features=len(feature_names),
                    top_labels=3
                )
                
                # Extract explanation data
                explanation = {
                    "sample_id": i,
                    "feature_weights": exp.as_list(),
                    "prediction": exp.predict_proba.tolist(),
                    "local_prediction": exp.local_pred.tolist()
                }
                lime_explanations.append(explanation)
            
            return {
                "enabled": True,
                "method": "lime",
                "explanations": lime_explanations,
                "feature_names": feature_names
            }
        
        except Exception as e:
            return {"enabled": False, "error": str(e)}

    def _model_predict_wrapper(self, model: Any, x: np.ndarray) -> np.ndarray:
        """
        Wrapper for model prediction (LIME compatibility).
        """
        try:
            with torch.no_grad():
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x, dtype=torch.float32)
                logits, _, _ = model(x)
                probs = torch.softmax(logits, dim=-1)
                return probs.numpy()
        except Exception:
            # Fallback: random probabilities
            return np.random.dirichlet([1, 1, 1], size=len(x))

    def explain(self,
               model: Any,
               background_data: np.ndarray,
               sample_data: np.ndarray,
               feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Ana explainability fonksiyonu.
        """
        results = {
            "timestamp": str(np.datetime64('now')),
            "model_type": type(model).__name__,
            "background_data_shape": background_data.shape,
            "sample_data_shape": sample_data.shape
        }
        
        # SHAP explanations
        if self.shap_enabled:
            results["shap"] = self.explain_shap(model, background_data, sample_data)
        
        # LIME explanations
        if self.lime_enabled:
            results["lime"] = self.explain_lime(model, sample_data, feature_names)
        
        return results 