from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database
    database_url: str

    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # File Upload
    upload_dir: str = "./uploads"
    max_file_size: int = 5242880  # 5MB

    class Config:
        env_file = ".env"


settings = Settings()