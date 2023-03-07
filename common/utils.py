import os
import io
import math
import re
import base64
import time
import arrow
import hashlib
import requests
import mimetypes
from config import Config
import threading
from typing import List, Dict, Callable, Optional
from datetime import datetime, timedelta
from pytz import timezone
from collections import defaultdict
from functools import wraps


class MarkdownUtils:

    @classmethod
    def extract_images(cls, text: str):
        # Check for headings
        pattern = r"!\[.*\]\((.*?)\)"
        return re.findall(pattern, text)


# 数字编码器
class IntEncoder:
    @classmethod
    def int_to_bytes(cls, x: int) -> bytes:
        return x.to_bytes((x.bit_length() + 7) // 8, 'big')

    @classmethod
    def int_from_bytes(cls, x_bytes: bytes) -> int:
        return int.from_bytes(x_bytes, 'big')

    @classmethod
    def encode_int(cls, i: int, prefix='int'):
        return prefix + base64.b64encode(cls.int_to_bytes(i)).decode('ascii').replace('=', '').replace('/', '').replace('+', '')

    @classmethod
    def encode_now(cls, prefix='time'):
        # 946684800000 -> 2000-01-01 00:00:00 UTC
        return cls.encode_int(int(time.time() * 1000), prefix=prefix)


# defaultdict 增强版
class BoostDict(defaultdict):
    def __init__(self, boost_factory):
        super().__init__(None)
        self.boost_factory = boost_factory

    def __missing__(self, key):
        if self.boost_factory is None:
            raise KeyError(key)
        self[key] = ret = self.boost_factory(key)
        return ret


class Fetcher:
    proxies = {'http': Config.proxy(), 'https': Config.proxy()} if Config.proxy() else None

    @classmethod
    def get(cls, url, params=None, **kwargs):
        return requests.get(url, params=params, proxies=cls.proxies, **kwargs)

    @classmethod
    def fetch_file_data(cls, url, retry_count=0) -> (io.BytesIO, str, str):
        try:
            response = requests.get(url, stream=True, proxies=cls.proxies)
            bytes_io = io.BytesIO()
            for chunk in response.iter_content(1024):
                bytes_io.write(chunk)
            mime_type = response.headers.get('Content-Type')
            file_type = mime_type.split('/')[0]
            file_ext = mimetypes.guess_extension(mime_type)
            bytes_io.seek(0)
            return bytes_io, file_type, file_ext
        except Exception as e:
            if retry_count < 2:
                time.sleep(1)
                return cls.fetch_file_data(url, retry_count+1)
            else:
                return None, '', ''
