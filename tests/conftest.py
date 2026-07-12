import os
import pytest

# Set env variables before any other imports or tests run
os.environ["DATABASE_URL"] = "sqlite:///main_test.db"
os.environ["GEMINI_API_KEY"] = "mock_gemini_api_key"
os.environ["LANGFUSE_SECRET_KEY"] = "mock_langfuse_secret_key"
os.environ["LANGFUSE_PUBLIC_KEY"] = "mock_langfuse_public_key"
os.environ["LANGFUSE_BASE_URL"] = "https://cloud.langfuse.com"
os.environ["OPEN_WEATHER_API_KEY"] = "mock_open_weather_api_key"
os.environ["AWS_ACCESS_KEY_ID"] = "mock_aws_access_key_id"
os.environ["AWS_SECRET_ACCESS_KEY"] = "mock_aws_secret_access_key"
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_S3_BUCKET"] = "mock-s3-bucket"
