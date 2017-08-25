__all__ = ['Hosts']

class Hosts(object):

    _paper1_complete_hosts = (166313, 147100, 165536, 61945, 132339, 149781, 33446, 150887)
    _paper1_incomplete_hosts = (161174, 85746, 145729, 140594, 126115, 13927, 137625, 129237)

    def __init__(self, database):
        self._database = database
        self._all_host_ids = self._database['hosts_flag0'].read()['NSAID'].tolist()

        t = self._database['hosts_named'].read()
        self._host_name_to_id = dict(zip((n.lower() for n in t['SAGA']), t['NSA']))
        self._host_id_to_name = dict(zip(t['NSA'], t['SAGA']))


    def resolve_id(self, hosts):
        try:
            hosts = int(hosts)
        except(TypeError, ValueError):
            pass
        else:
            return [hosts]

        try:
            hosts = hosts.lower()
        except AttributeError:
            pass
        else:
            if hosts == 'all':
                return self._all_host_ids
            elif hosts == 'paper1_complete':
                return list(self._paper1_complete_hosts)
            elif hosts == 'paper1_incomplete':
                return list(self._paper1_incomplete_hosts)
            elif hosts == 'paper1':
                return list(self._paper1_complete_hosts + self._paper1_incomplete_hosts)
            elif hosts.lower() in self._host_name_to_id:
                return [self._host_name_to_id[hosts.lower()]]
            else:
                raise ValueError('cannot resolve {}'.format(hosts))

        out = []
        for host in hosts:
            out.extend(self.resolve_id(host))
        return out


    def id_to_name(self, host_id):
        return self._host_id_to_name.get(host_id)


    def load(self, host_type='flag0', reload=False):
        return self._database['hosts_{}'.format(host_type)].read(reload=reload)
