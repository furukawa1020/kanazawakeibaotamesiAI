"""Model registry for versioning and management."""
from pathlib import Path
from typing import Optional, Dict, Any
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Simple model registry for versioning."""
    
    def __init__(self, registry_dir: str = "models"):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory to store models
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"
        
        # Load or create registry
        if self.registry_file.exists():
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                self.registry = json.load(f)
        else:
            self.registry = {'models': {}}
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register a model.
        
        Args:
            model_name: Name/version of the model
            model_path: Path to model file
            metadata: Additional metadata
        """
        model_info = {
            'path': str(model_path),
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.registry['models'][model_name] = model_info
        self._save_registry()
        
        logger.info(f"Registered model: {model_name}")
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """Get path to a registered model."""
        if model_name in self.registry['models']:
            return self.registry['models'][model_name]['path']
        return None
    
    def list_models(self) -> list:
        """List all registered models."""
        return list(self.registry['models'].keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a model."""
        return self.registry['models'].get(model_name)
    
    def _save_registry(self):
        """Save registry to disk."""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)
