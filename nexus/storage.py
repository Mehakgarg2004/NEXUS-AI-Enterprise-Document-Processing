"""File storage: local disk or AWS S3."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from werkzeug.utils import secure_filename

from .config import Config


class LocalStorage:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, data: bytes, filename: str) -> str:
        key = f"{uuid.uuid4().hex}_{secure_filename(filename) or 'file'}"
        (self.root / key).write_bytes(data)
        return key

    def load(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def delete(self, key: str) -> None:
        try:
            os.remove(self.root / key)
        except FileNotFoundError:
            pass


class S3Storage:
    def __init__(self, bucket: str, region: str):
        import boto3

        self.bucket = bucket
        self.s3 = boto3.client("s3", region_name=region)

    def save(self, data: bytes, filename: str) -> str:
        key = f"documents/{uuid.uuid4().hex}_{secure_filename(filename) or 'file'}"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ServerSideEncryption="AES256")
        return key

    def load(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def delete(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=key)


def get_storage():
    if Config.STORAGE_BACKEND == "s3":
        if not Config.S3_BUCKET:
            raise RuntimeError("STORAGE_BACKEND=s3 requires S3_BUCKET")
        return S3Storage(Config.S3_BUCKET, Config.AWS_REGION)
    return LocalStorage(Config.UPLOAD_DIR)
