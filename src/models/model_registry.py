"""Model Registry — Version management for deployed models."""

from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass
class ModelVersion:
    version: str
    model_type: str  # trocr, vlm
    checkpoint_path: str
    metrics: dict
    is_champion: bool = False


class ModelRegistry:
    """Manages model versions and champion selection."""

    def __init__(self):
        self.versions: list[ModelVersion] = []
        self._champion: ModelVersion | None = None
        self.is_loaded = False
        self.champion_version: str | None = None

    async def load_champion_model(self):
        """Load the current champion model."""
        # In production, this would load from model store
        self.is_loaded = True
        self.champion_version = "v0.1.0-baseline"
        logger.info("registry.champion_loaded", version=self.champion_version)

    async def unload_models(self):
        """Unload all models from memory."""
        self.is_loaded = False
        logger.info("registry.unloaded")

    def register_model(self, version: str, model_type: str, checkpoint_path: str, metrics: dict):
        """Register a new model version."""
        model = ModelVersion(
            version=version,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            metrics=metrics,
        )
        self.versions.append(model)
        logger.info("registry.registered", version=version, metrics=metrics)

    def promote_to_champion(self, version: str) -> bool:
        """Promote a model version to champion (production)."""
        for model in self.versions:
            if model.version == version:
                # Demote current champion
                if self._champion:
                    self._champion.is_champion = False
                model.is_champion = True
                self._champion = model
                self.champion_version = version
                logger.info("registry.promoted", version=version)
                return True
        return False
