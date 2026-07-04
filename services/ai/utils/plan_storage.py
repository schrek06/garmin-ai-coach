import logging
from abc import ABC, abstractmethod
from pathlib import Path

from services.db.database import load_latest_training_plan, save_training_plan

logger = logging.getLogger(__name__)


class PlanStorage(ABC):

    @abstractmethod
    def load_plan(self, user_id: str, plan_type: str) -> str | None:
        pass

    @abstractmethod
    def save_plan(self, user_id: str, plan_type: str, content: str) -> None:
        pass

class FilePlanStorage(PlanStorage):
    # Keeping class name FilePlanStorage for compatibility but using DB backend

    def __init__(self, base_dir: str = "data/storage"):
        pass

    def load_plan(self, user_id: str, plan_type: str) -> str | None:
        try:
            return load_latest_training_plan(user_id, plan_type)
        except Exception:
            logger.exception("Unexpected error loading %s for user %s", plan_type, user_id)
            return None

    def save_plan(self, user_id: str, plan_type: str, content: str) -> None:
        if not content:
            logger.warning("Attempted to save empty content for %s (user: %s)", plan_type, user_id)
            return
        
        try:
            save_training_plan(user_id, plan_type, content)
        except Exception:
            logger.exception("Unexpected error saving %s for user %s", plan_type, user_id)
