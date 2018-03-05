"""Mongo Interface."""
from hsdbi import mongo


class MongoDbInterface(mongo.MongoDbFacade):
    """Interface for access to MongoDB."""

    def __init__(self, db_name):
        super(MongoDbInterface, self).__init__(db_name=db_name)
        self.experiments = mongo.MongoRepository(self.db, 'experiments')
        self.grid = mongo.MongoRepository(self.db, 'grid')
        self.histories = mongo.MongoRepository(self.db, 'histories')

