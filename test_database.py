import os
import tempfile
import numpy as np
from SAGA.database import Database, DataObject, CsvTable, FitsTable


def safe_array_equal(a1, a2):
    return np.array_equal(a1, a2) or np.array_equal(a1, a2, equal_nan=True)


def test_create_database():
    _ = Database()
    _ = Database(".")


def test_download_google():
    d = Database()
    d1 = d["hosts"].read()
    t = tempfile.mkstemp()[1]
    try:
        d["hosts"].download(t, overwrite=True)
        d["hosts"].clear_cache()
        d["hosts"].remote = None
        d2 = d["hosts"].read()
        for k in d1.columns:
            assert safe_array_equal(d1[k], d2[k])
    finally:
        os.unlink(t)


def test_set_local():
    d = Database()
    d1 = d["hosts"].read()
    t = tempfile.mkstemp()[1]
    try:
        d["hosts"].download(t, overwrite=True, set_as_local=False)
        d["hosts"].clear_cache()
        d["hosts"].remote = None
        d["hosts"].local = t
        d2 = d["hosts"].read()
        for k in d1.columns:
            assert safe_array_equal(d1[k], d2[k])
    finally:
        os.unlink(t)


def test_set_base():
    d = Database()
    d["base", 32].local = "path"
    d1 = d.get(("base", 32))
    try:
        d1.read()
    except FileNotFoundError:
        pass
    assert isinstance(d1.local, FitsTable)
    assert d1.local.path == "path"


def test_download_default():
    d = Database()
    d1 = d["hosts"].read()
    t = tempfile.mkstemp()[1]
    try:
        dobj = DataObject(d["hosts"].remote, CsvTable(t))
        dobj.download(overwrite=True)
        dobj.clear_cache()
        dobj.remote = None
        d2 = dobj.read()
        for k in d1.columns:
            assert safe_array_equal(d1[k], d2[k])
    finally:
        os.unlink(t)
