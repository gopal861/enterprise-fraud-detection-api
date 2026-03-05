from pydantic import BaseModel
import os


def get_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class Settings(BaseModel):
    db_name: str
    db_user: str
    db_password: str
    db_host: str
    db_port: str
    model_version_id: int
    openai_api_key: str


settings = Settings(
    db_name=get_env("DB_NAME"),
    db_user=get_env("DB_USER"),
    db_password=get_env("DB_PASSWORD"),
    db_host=get_env("DB_HOST"),
    db_port=get_env("DB_PORT"),
    model_version_id=int(get_env("MODEL_VERSION_ID")),
    openai_api_key=get_env("OPENAI_API_KEY"),
)