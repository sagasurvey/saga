from SAGA.database import *
import tempfile
import os
import numpy as np

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

def test_create_database():
    d = Database()
    d = Database('.')

def test_download_google():
    d = Database()
    d1 = d['hosts_named'].read()
    t = tempfile.mkstemp()[1]
    try:
        d['hosts_named'].download(t, overwrite=True)
        d['hosts_named'].clear_cache()
        d['hosts_named'].remote = None
        d2 = d['hosts_named'].read()
        for k in d1.columns:
            assert (d1[k]==d2[k]).all()
    finally:
        os.unlink(t)

def test_set_local():
    d = Database()
    d1 = d['hosts_named'].read()
    t = tempfile.mkstemp()[1]
    try:
        d['hosts_named'].download(t, overwrite=True, set_as_local=False)
        d['hosts_named'].clear_cache()
        d['hosts_named'].remote = None
        d['hosts_named'].local = t
        d2 = d['hosts_named'].read()
        for k in d1.columns:
            assert (d1[k]==d2[k]).all()
    finally:
        os.unlink(t)

def test_set_base():
    d = Database()
    d['base', 32].local = 'path'
    d1 = d.get(('base', 32))
    try:
        d1.read()
    except FileNotFoundError:
        pass
    assert isinstance(d1.local, FitsTable)
    assert d1.local.path == 'path'

def test_download_default():
    d = Database()
    d1 = d['hosts_named'].read()
    t = tempfile.mkstemp()[1]
    try:
        dobj = DataObject(d['hosts_named'].remote, CsvTable(t))
        dobj.download(overwrite=True)
        dobj.clear_cache()
        dobj.remote = None
        d2 = dobj.read()
        for k in d1.columns:
            assert (d1[k]==d2[k]).all()
    finally:
        os.unlink(t)
