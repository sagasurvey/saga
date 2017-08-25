import numpy as np
from astropy.table import vstack
from easyquery import Query
from . import queries
from ..hosts import Hosts

__all__ = ['Objects']


def _slice_columns(table, columns):
    return table[columns] if columns is not None else table


class Objects(object):

    def __init__(self, database):
        self._database = database
        self._hosts = Hosts(self._database)


    @staticmethod
    def _add_colors(table):
        sdss_bands = 'ugriz'
        for b in sdss_bands:
            table['{}_mag'.format(b)] = table[b] - table['EXTINCTION_{}'.format(b.upper())]

        for color in map(''.join, zip(sdss_bands[:-1], sdss_bands[1:])):
            table[color] = table['{}_mag'.format(color[0])] - table['{}_mag'.format(color[1])]
            table['{}_err'.format(color)] = np.sqrt(table['{}_err'.format(color[0])]**2.0 + table['{}_err'.format(color[1])]**2.0)

        return table




    def load(self, hosts=None, has_spec=None, query=None, iter_hosts=False, columns=None):
        if has_spec:
            t = self._database['spectra_clean'].read()

            if hosts is not None:
                host_ids = self._hosts.resolve_id(hosts)
                t = Query((lambda x: np.in1d(x, host_ids), 'HOST_NSAID')).filter(t)

            t = self._add_colors(t)

            if query is not None:
                t = Query(query).filter(t)

            if iter_hosts:
                if hosts is None:
                    host_ids = np.unique(t['HOST_NSAID'])
                return (Query('HOST_NSAID == {}'.format(i)).filter(t) for i in host_ids)
            else:
                return _slice_columns(t, columns)

        else:
            q = Query(query)
            if has_spec is not None:
                q = q & (~queries.has_spec)

            hosts = self._hosts.resolve_id('all') if hosts is None else self._hosts.resolve_id(hosts)

            output_iterator = (_slice_columns(q.filter(self._add_colors(self._database['base', host].read())), columns) for host in hosts)

            return output_iterator if iter_hosts else vstack(list(output_iterator))


    def build(self, hosts=None, rebuild=False):
        raise NotImplementedError #TODO: implement this
