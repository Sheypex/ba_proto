import atexit
import dataclasses

import my_yaml
import data_types
from pprint import pprint
import psycopg2

import sqlalchemy as sqlA


class _db_actions:
    singleton: '_db_actions' = None

    @staticmethod
    def _getInstance():
        if _db_actions.singleton is None:
            _db_actions.singleton = _db_actions()
        return _db_actions.singleton

    def __init__(self):
        if _db_actions.singleton is None:  # singleton
            # setup
            yaml_config = my_yaml.pgconfig
            self.pg_creds = yaml_config['postgres_credentials']
            # connect
            self._engine = None
            # singleton
            _db_actions.singleton = self
        else:
            raise RuntimeError('This class is a singleton. It shouldn\'t be instantiated directly.')

    def engine(self):
        if self._engine is None:
            self._engine = sqlA.create_engine(
                f"postgresql+psycopg2://{self.pg_creds['user']}:{self.pg_creds['password']}@{self.pg_creds['host']}:{self.pg_creds['port']}/{self.pg_creds['dbname']}")
        return self._engine

    def connect(self):
        return self.engine().connect()

    def begin(self):
        return self.engine().begin()

    def exec(self, sql, params=None, func=None):
        with self.engine().connect() as conn:
            if params:
                res = conn.execute(sqlA.text(sql), params).fetchall()
            else:
                res = conn.execute(sqlA.text(sql)).fetchall()
            if func:
                return func(res)
            else:
                return res

    def execB(self, sql, params=None, func=None):
        with self.engine().begin() as conn:
            if params:
                res = conn.execute(sqlA.text(sql), params).fetchall()
            else:
                res = conn.execute(sqlA.text(sql)).fetchall()
            if func:
                return func(res)
            else:
                return res


db_actions = _db_actions._getInstance()
