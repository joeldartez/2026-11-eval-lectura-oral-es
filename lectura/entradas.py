"""Fuentes de archivos. R2 conserva los bytes y el nombre original del audio."""

import os
from contextlib import contextmanager
from itertools import islice
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory


EXTENSIONS = {".wav", ".ogg", ".flac", ".mp3", ".aif", ".aiff", ".opus"}


def locales(folder):
    return (p for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in EXTENSIONS)


class R2:
    def __init__(self):
        import boto3

        names = ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
        missing = [k for k in names if not os.environ.get(k)]
        if missing:
            raise ValueError("Faltan variables de R2: " + ", ".join(missing))
        endpoint, access, secret, self.bucket = (os.environ[k] for k in names)
        self.client = boto3.client("s3", endpoint_url=endpoint, region_name="auto",
                                   aws_access_key_id=access, aws_secret_access_key=secret)

    def claves(self, prefix, limit):
        pages = self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix)
        keys = (obj["Key"] for page in pages for obj in page.get("Contents", [])
                if Path(obj["Key"]).suffix.lower() in EXTENSIONS)
        return islice(keys, limit)

    @contextmanager
    def audio(self, key, work):
        name = key.rsplit("/", 1)[-1]
        if not name or name in {".", ".."} or PureWindowsPath(name).name != name or ":" in name:
            raise ValueError(f"Nombre de audio no portable: {key!r}")
        work.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=work) as tmp:
            path = Path(tmp) / name
            self.client.download_file(self.bucket, key, str(path))
            yield path

    def publicar(self, path, root, prefix):
        key = "/".join(filter(None, (prefix.strip("/"), path.relative_to(root).as_posix())))
        self.client.upload_file(str(path), self.bucket, key)
        print(f"s3://{self.bucket}/{key}")
