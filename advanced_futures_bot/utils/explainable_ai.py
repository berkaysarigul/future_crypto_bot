import logging
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available, explainability limited")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available, explainability limited")

logger = logging.getLogger("ExplainableAI")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class ExplainableAI:
    """
    Production-level model explainability and interpretability.
    - SHAP and LIME explanations
    - Feature importance analysis
    - Model interpretability tools
    - Exception handling, logging
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.output_dir = self.config.get("explainability_dir", "explainability")
        self.feature_names = self.config.get("feature_names", [])
        self.class_names = self.config.get("class_names", ["sell", "hold", "buy"])
        self.max_samples = self.config.get("max_explanation_samples", 100)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        logger.info("ExplainableAI initialized successfully")

    def setup_shap_explainer(self, model, background_data: np.ndarray):
        """Setup SHAP explainer with background data."""
        try:
            if not SHAP_AVAILABLE:
                logger.warning("SHAP not available, skipping explainer setup")
                return False
            
            if background_data.shape[0] > self.max_samples:
                # Sample background data to avoid memory issues
                indices = np.random.choice(background_data.shape[0], self.max_samples, replace=False)
                background_data = background_data[indices]
                logger.info(f"Sampled {self.max_samples} background samples for SHAP")
            
            # Choose appropriate explainer based on model type
            if hasattr(model, 'predict_proba'):
                self.shap_explainer = shap.KernelExplainer(model.predict_proba, background_data)
            elif hasattr(model, 'predict'):
                self.shap_explainer = shap.KernelExplainer(model.predict, background_data)
            else:
                logger.error("Model does not have predict or predict_proba method")
                return False
            
            logger.info("SHAP explainer setup successfully")
            return True
            
        except Exception as e:
            logger.error(f"SHAP explainer setup error: {e}")
            return False

    def setup_lime_explainer(self, training_data: np.ndarray):
        """Setup LIME explainer with training data."""
        try:
            if not LIME_AVAILABLE:
                logger.warning("LIME not available, skipping explainer setup")
                return False
            
            if len(self.feature_names) != training_data.shape[1]:
                self.feature_names = [f"feature_{i}" for i in range(training_data.shape[1])]
                logger.warning(f"Feature names mismatch, using default names")
            
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification'
            )
            
            logger.info("LIME explainer setup successfully")
            return True
            
        except Exception as e:
            logger.error(f"LIME explainer setup error: {e}")
            return False

    def explain_prediction_shap(self, 
                               model, 
                               sample: np.ndarray, 
                               background_data: np.ndarray = None) -> Dict[str, Any]:
        """Generate SHAP explanation for a single prediction."""
        try:
            if not SHAP_AVAILABLE:
                return {"error": "SHAP not available"}
            
            if self.shap_explainer is None:
                if background_data is None:
                    return {"error": "No background data provided for SHAP"}
                self.setup_shap_explainer(model, background_data)
            
            if self.shap_explainer is None:
                return {"error": "SHAP explainer setup failed"}
            
            # Generate SHAP values
            shap_values = self.shap_explainer.shap_values(sample)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)
            
            # Get feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            explanation = {
                "shap_values": shap_values.tolist(),
                "feature_importance": feature_importance.tolist(),
                "feature_names": self.feature_names,
                "prediction": model.predict(sample)[0] if hasattr(model, 'predict') else None,
                "prediction_proba": model.predict_proba(sample)[0] if hasattr(model, 'predict_proba') else None
            }
            
            logger.info(f"SHAP explanation generated for sample")
            return explanation
            
        except Exception as e:
            logger.error(f"SHAP explanation error: {e}")
            return {"error": str(e)}

    def explain_prediction_lime(self, 
                               model, 
                               sample: np.ndarray, 
                               num_features: int = 10) -> Dict[str, Any]:
        """Generate LIME explanation for a single prediction."""
        try:
            if not LIME_AVAILABLE:
                return {"error": "LIME not available"}
            
            if self.lime_explainer is None:
                return {"error": "LIME explainer not setup"}
            
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                sample[0],  # LIME expects 1D array
                model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                num_features=num_features
            )
            
            # Extract explanation data
            lime_data = {
                "feature_weights": dict(explanation.as_list()),
                "prediction": explanation.predicted_class,
                "confidence": explanation.score,
                "feature_names": self.feature_names
            }
            
            logger.info(f"LIME explanation generated for sample")
            return lime_data
            
        except Exception as e:
            logger.error(f"LIME explanation error: {e}")
            return {"error": str(e)}

    def analyze_feature_importance(self, 
                                  model, 
                                  X: np.ndarray, 
                                  y: np.ndarray = None) -> Dict[str, Any]:
        """Analyze feature importance using multiple methods."""
        try:
            importance_analysis = {}
            
            # Method 1: SHAP-based importance
            if SHAP_AVAILABLE and self.shap_explainer is not None:
                try:
                    # Sample data for SHAP analysis
                    sample_size = min(100, X.shape[0])
                    sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
                    X_sample = X[sample_indices]
                    
                    shap_values = self.shap_explainer.shap_values(X_sample)
                    if isinstance(shap_values, list):
                        shap_values = np.array(shap_values)
                    
                    shap_importance = np.abs(shap_values).mean(axis=0)
                    importance_analysis["shap_importance"] = {
                        "values": shap_importance.tolist(),
                        "feature_names": self.feature_names
                    }
                except Exception as e:
                    logger.warning(f"SHAP importance analysis failed: {e}")
            
            # Method 2: Permutation importance (if sklearn available)
            try:
                from sklearn.inspection import permutation_importance
                from sklearn.model_selection import train_test_split
                
                if y is not None:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    perm_importance = permutation_importance(
                        model, X_test, y_test, n_repeats=10, random_state=42
                    )
                    importance_analysis["permutation_importance"] = {
                        "values": perm_importance.importances_mean.tolist(),
                        "std": perm_importance.importances_std.tolist(),
                        "feature_names": self.feature_names
                    }
            except ImportError:
                logger.info("sklearn not available for permutation importance")
            except Exception as e:
                logger.warning(f"Permutation importance analysis failed: {e}")
            
            # Method 3: Model-specific importance
            if hasattr(model, 'feature_importances_'):
                importance_analysis["model_importance"] = {
                    "values": model.feature_importances_.tolist(),
                    "feature_names": self.feature_names
                }
            
            logger.info("Feature importance analysis completed")
            return importance_analysis
            
        except Exception as e:
            logger.error(f"Feature importance analysis error: {e}")
            return {"error": str(e)}

    def create_explanation_plots(self, 
                                explanations: List[Dict[str, Any]], 
                                save_plots: bool = True) -> Dict[str, str]:
        """Create visualization plots for explanations."""
        try:
            plot_paths = {}
            
            # Feature importance plot
            if explanations and "feature_importance" in explanations[0]:
                plt.figure(figsize=(12, 8))
                
                importance_data = explanations[0]["feature_importance"]
                feature_names = explanations[0].get("feature_names", [f"Feature_{i}" for i in range(len(importance_data))])
                
                # Sort by importance
                sorted_indices = np.argsort(importance_data)[::-1]
                sorted_importance = [importance_data[i] for i in sorted_indices]
                sorted_names = [feature_names[i] for i in sorted_indices]
                
                plt.barh(range(len(sorted_importance)), sorted_importance)
                plt.yticks(range(len(sorted_names)), sorted_names)
                plt.xlabel("Feature Importance")
                plt.title("Feature Importance Analysis")
                plt.tight_layout()
                
                if save_plots:
                    plot_path = os.path.join(self.output_dir, "feature_importance.png")
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plot_paths["feature_importance"] = plot_path
                
                plt.close()
            
            # SHAP summary plot
            if SHAP_AVAILABLE and explanations and "shap_values" in explanations[0]:
                try:
                    plt.figure(figsize=(12, 8))
                    
                    shap_values = np.array(explanations[0]["shap_values"])
                    feature_names = explanations[0].get("feature_names", [f"Feature_{i}" for i in range(shap_values.shape[1])])
                    
                    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
                    
                    if save_plots:
                        plot_path = os.path.join(self.output_dir, "shap_summary.png")
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plot_paths["shap_summary"] = plot_path
                    
                    plt.close()
                except Exception as e:
                    logger.warning(f"SHAP summary plot failed: {e}")
            
            # LIME explanation plot
            if explanations and "feature_weights" in explanations[0]:
                plt.figure(figsize=(10, 6))
                
                weights = explanations[0]["feature_weights"]
                features = list(weights.keys())
                values = list(weights.values())
                
                # Sort by absolute value
                sorted_indices = np.argsort(np.abs(values))
                sorted_features = [features[i] for i in sorted_indices]
                sorted_values = [values[i] for i in sorted_indices]
                
                colors = ['red' if v < 0 else 'green' for v in sorted_values]
                plt.barh(range(len(sorted_values)), sorted_values, color=colors)
                plt.yticks(range(len(sorted_features)), sorted_features)
                plt.xlabel("Feature Weight")
                plt.title("LIME Feature Weights")
                plt.tight_layout()
                
                if save_plots:
                    plot_path = os.path.join(self.output_dir, "lime_explanation.png")
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plot_paths["lime_explanation"] = plot_path
                
                plt.close()
            
            logger.info(f"Explanation plots created: {list(plot_paths.keys())}")
            return plot_paths
            
        except Exception as e:
            logger.error(f"Explanation plots creation error: {e}")
            return {"error": str(e)}

    def generate_explanation_report(self, 
                                   model, 
                                   X: np.ndarray, 
                                   y: np.ndarray = None,
                                   sample_indices: List[int] = None) -> Dict[str, Any]:
        """Generate comprehensive explanation report."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "model_type": type(model).__name__,
                "data_shape": X.shape,
                "feature_names": self.feature_names,
                "explanations": [],
                "feature_importance": {},
                "plots": {}
            }
            
            # Sample data for explanations
            if sample_indices is None:
                sample_size = min(10, X.shape[0])
                sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
            
            # Generate explanations for samples
            for i, idx in enumerate(sample_indices):
                sample = X[idx:idx+1]
                
                # SHAP explanation
                shap_explanation = self.explain_prediction_shap(model, sample, X)
                if "error" not in shap_explanation:
                    report["explanations"].append({
                        "sample_index": idx,
                        "shap": shap_explanation
                    })
                
                # LIME explanation
                lime_explanation = self.explain_prediction_lime(model, sample)
                if "error" not in lime_explanation:
                    if report["explanations"]:
                        report["explanations"][-1]["lime"] = lime_explanation
                    else:
                        report["explanations"].append({
                            "sample_index": idx,
                            "lime": lime_explanation
                        })
            
            # Feature importance analysis
            importance_analysis = self.analyze_feature_importance(model, X, y)
            if "error" not in importance_analysis:
                report["feature_importance"] = importance_analysis
            
            # Create plots
            if report["explanations"]:
                plot_paths = self.create_explanation_plots(report["explanations"])
                if "error" not in plot_paths:
                    report["plots"] = plot_paths
            
            # Save report
            report_path = os.path.join(self.output_dir, f"explanation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            report["report_path"] = report_path
            logger.info(f"Explanation report generated: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"Explanation report generation error: {e}")
            return {"error": str(e)}

    def explain_model_decision(self, 
                              model, 
                              state: np.ndarray, 
                              action: int = None) -> Dict[str, Any]:
        """Explain model decision for a specific state."""
        try:
            explanation = {
                "state": state.tolist(),
                "action": action,
                "timestamp": datetime.now().isoformat()
            }
            
            # Get model prediction
            if hasattr(model, 'predict'):
                prediction = model.predict(state.reshape(1, -1))[0]
                explanation["prediction"] = prediction
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(state.reshape(1, -1))[0]
                explanation["probabilities"] = probabilities.tolist()
            
            # Generate SHAP explanation
            shap_explanation = self.explain_prediction_shap(model, state.reshape(1, -1))
            if "error" not in shap_explanation:
                explanation["shap"] = shap_explanation
            
            # Generate LIME explanation
            lime_explanation = self.explain_prediction_lime(model, state.reshape(1, -1))
            if "error" not in lime_explanation:
                explanation["lime"] = lime_explanation
            
            logger.info(f"Model decision explanation generated")
            return explanation
            
        except Exception as e:
            logger.error(f"Model decision explanation error: {e}")
            return {"error": str(e)} 