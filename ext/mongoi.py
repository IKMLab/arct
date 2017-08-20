"""MongoDB interface."""
from hsdbi import mongo


class Facade(mongo.MongoFacade):
    """Meta-Facade for all db access.
    """

    def __init__(self, server='localhost', port=27017):
        super(Facade, self).__init__(server, port)
        self.history = mongo.MongoDbFacade(
            self.connection,
            db_name='history',
            collections=['train'])