from common.data import Reply, Context, Query


class Channel(object):
    def startup(self):
        """
        startup channel
        """
        raise NotImplementedError
