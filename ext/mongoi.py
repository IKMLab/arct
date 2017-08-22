"""MongoDB interface."""
from hsdbi import mongo


class DbInterface(mongo.MongoFacade):
    """For access to MongoDB for saving and loading histories."""

    def __init__(self, server='localhost', port=27017):
        super(DbInterface, self).__init__(server, port)
        self.history = mongo.MongoDbFacade(
            self.connection,
            db_name='history',
            collections=['train'])
