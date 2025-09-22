import random
from typing import Literal

# Categories: safe / nudity / violence / disallowed
ModerationResult = Literal["safe", "nudity", "violence", "disallowed"]

class ModerationService:
    @staticmethod
    def analyze_image(image_bytes: bytes) -> ModerationResult:
        """
        Mock moderation: randomly flags content.
        Later: replace with TensorFlow Lite EfficientNet.
        """
        outcomes = ["safe", "nudity", "violence", "disallowed"]
        # 80% safe, 20% unsafe (for testing)
        result = random.choices(outcomes, weights=[0.8, 0.1, 0.05, 0.05])[0]
        return result
