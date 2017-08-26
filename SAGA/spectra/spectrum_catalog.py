class SpectrumCatalog(object):

    def __init__(self, database):
        self._database = database

    def load(self, telescope=None):
        raise NotImplementedError #TODO: implement this
