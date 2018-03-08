"""Interface to MongoDB."""
from ext import mongoi
from hsdbi import mongo


class MongoDbInterface(mongoi.MongoDbInterface):
    """Interface for access to MongoDB."""

    def __init__(self):
        super(MongoDbInterface, self).__init__(db_name='arctr')
        self.arct = mongo.MongoRepository(db=self.db, collection_name='arct')
