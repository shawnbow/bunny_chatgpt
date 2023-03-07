# encoding:utf-8
import os
import json
from pprint import pprint


class Config:
    workdir = os.path.dirname(os.path.abspath(__file__))
    config_path = f'{workdir}/config.json'

    config = dict()

    @classmethod
    def load(cls):
        if not os.path.exists(cls.config_path):
            raise Exception('配置文件不存在, 请根据config-template.json模板创建config.json文件')
        try:
            with open(cls.config_path, 'r') as f:
                cls.config = json.load(f)
        except Exception as e:
            raise Exception('读取配置文件失败, 请根据config-template.json模板创建config.json文件')

    @classmethod
    def save(cls):
        try:
            with open(cls.config_path, 'w') as f:
                json.dump(cls.config, f, indent=4)
        except Exception as e:
            raise Exception('保存配置文件失败!')

    @classmethod
    def set(cls, key, value):
        cls.config[key] = value
        cls.save()

    @classmethod
    def get(cls, key=None, default=None):
        if not key:
            return cls.config
        return cls.config.get(key, default)

    @classmethod
    def db_path(cls):
        return f'{cls.workdir}/{cls.get("db_path", "db")}'

    @classmethod
    def log_path(cls):
        return f'{cls.workdir}/{cls.get("log_path", "logs")}'

    @classmethod
    def proxy(cls):
        return cls.get('proxy')

    @classmethod
    def openai(cls, key=None):
        if not key:
            return cls.get('openai', {})
        return cls.get('openai', {}).get(key, None)

    @classmethod
    def gradio(cls, key=None):
        if not key:
            return cls.get('gradio', {})
        return cls.get('gradio', {}).get(key, None)


Config.load()


if __name__ == "__main__":
    pprint(Config.get())
