"""
MongoDB database configuration and management for Cameroon agricultural system
Centralized class for all database operations
"""

import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT, GEOSPHERE
from pymongo.errors import ConnectionFailure
from bson import ObjectId
from bson.errors import InvalidId

from .settings import Settings, get_settings

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """MongoDB connection configuration and manager for agricultural data"""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._client: Optional[AsyncIOMotorClient] = None
        self._database: Optional[AsyncIOMotorDatabase] = None
        self._sync_client: Optional[MongoClient] = None
        self._connection_verified = False
        self._indexes_created = False

        self.collections = {
            "soil_data": self.settings.get_collection_name("soil_data"),
            "weather_data": self.settings.get_collection_name("weather_data"),
            "crop_data": self.settings.get_collection_name("crop_data"),
            "satellite_data": self.settings.get_collection_name("satellite_data"),
            "field_metadata": self.settings.get_collection_name("field_metadata"),
            "yield_data": self.settings.get_collection_name("yield_data"),
            "irad_data": self.settings.get_collection_name("irad_data"),
            "synthetic_data": self.settings.get_collection_name("synthetic_data"),
            "model_predictions": self.settings.get_collection_name("model_predictions"),
            "data_quality_logs": self.settings.get_collection_name("data_quality_logs"),
            "user_sessions": self.settings.get_collection_name("user_sessions"),
            "system_metrics": self.settings.get_collection_name("system_metrics"),
        }

    @property
    def client(self) -> Optional[AsyncIOMotorClient]:
        """Public access to the async MongoDB client"""
        return self._client

    async def connect(self) -> AsyncIOMotorDatabase:
        """Establishes MongoDB connection and returns database"""
        if self._database and self._connection_verified:
            return self._database

        try:
            logger.info("Connecting to MongoDB: %s", self.settings.mongodb_database)

            self._client = AsyncIOMotorClient(
                self.settings.mongodb_connection_string,
                maxPoolSize=self.settings.mongodb_max_pool_size,
                minPoolSize=self.settings.mongodb_min_pool_size,
                socketTimeoutMS=self.settings.mongodb_socket_timeout_ms,
                connectTimeoutMS=self.settings.mongodb_connect_timeout_ms,
                serverSelectionTimeoutMS=self.settings.mongodb_server_selection_timeout_ms,
                retryWrites=True,
                retryReads=True,
            )

            self._database = self._client[self.settings.mongodb_database]
            await self._verify_connection()
            if not self._indexes_created:
                await self._setup_indexes()
                self._indexes_created = True

            logger.info("MongoDB connection established successfully")
            return self._database

        except Exception as e:
            logger.error("Failed to connect to MongoDB: %s", e)
            raise ConnectionFailure(f"MongoDB connection failed: {e}")

    async def disconnect(self) -> None:
        """Closes the MongoDB connection"""
        if self._client:
            self._client.close()
            self._client = None
            self._database = None
            self._connection_verified = False
            logger.info("MongoDB connection closed")

    async def _verify_connection(self) -> None:
        """Verifies the MongoDB connection is working"""
        try:
            await self._client.admin.command("ping")
            self._connection_verified = True
            logger.info("MongoDB connection verified")
        except Exception as e:
            self._connection_verified = False
            raise ConnectionFailure(f"MongoDB connection verification failed: {e}")

    async def _setup_indexes(self) -> None:
        """Configures MongoDB indexes for performance optimization"""
        try:
            logger.info("Setting up MongoDB indexes...")

            # Spatial indexes for all geo-referenced collections
            spatial_collections = ["soil_data", "weather_data", "crop_data", "field_metadata"]
            for coll_name in spatial_collections:
                collection = self._database[self.collections[coll_name]]
                await collection.create_index([("coordinates", GEOSPHERE)])
                await collection.create_index([("latitude", ASCENDING), ("longitude", ASCENDING)])
                await collection.create_index([("date", DESCENDING)])
                await collection.create_index([("created_at", DESCENDING)])
                await collection.create_index([
                    ("coordinates", GEOSPHERE),
                    ("date", DESCENDING),
                ])

            # Soil data indexes
            soil_collection = self._database[self.collections["soil_data"]]
            await soil_collection.create_index([("field_id", ASCENDING)])
            await soil_collection.create_index([("ph_water", ASCENDING)])
            await soil_collection.create_index([("organic_carbon", ASCENDING)])
            await soil_collection.create_index([("texture_class", ASCENDING)])

            # Weather data indexes
            weather_collection = self._database[self.collections["weather_data"]]
            await weather_collection.create_index([("station_id", ASCENDING)])
            await weather_collection.create_index([("temperature_avg", ASCENDING)])
            await weather_collection.create_index([("precipitation_daily", ASCENDING)])
            await weather_collection.create_index([
                ("station_id", ASCENDING),
                ("date", DESCENDING),
            ])

            # Crop data indexes
            crop_collection = self._database[self.collections["crop_data"]]
            await crop_collection.create_index([("crop_type", ASCENDING)])
            await crop_collection.create_index([("variety", ASCENDING)])
            await crop_collection.create_index([("season", ASCENDING)])
            await crop_collection.create_index([("yield_tha", ASCENDING)])
            await crop_collection.create_index([
                ("crop_type", ASCENDING),
                ("season", ASCENDING),
                ("year", DESCENDING),
            ])

            # Field metadata indexes
            field_collection = self._database[self.collections["field_metadata"]]
            await field_collection.create_index([("field_id", ASCENDING)], unique=True)
            await field_collection.create_index([("agroecological_zone", ASCENDING)])
            await field_collection.create_index([("farm_size_category", ASCENDING)])

            # Satellite data indexes
            satellite_collection = self._database[self.collections["satellite_data"]]
            await satellite_collection.create_index([("satellite_mission", ASCENDING)])
            await satellite_collection.create_index([("processing_level", ASCENDING)])
            await satellite_collection.create_index([("cloud_cover", ASCENDING)])
            await satellite_collection.create_index([
                ("satellite_mission", ASCENDING),
                ("acquisition_date", DESCENDING),
            ])

            # Model predictions indexes
            prediction_collection = self._database[self.collections["model_predictions"]]
            await prediction_collection.create_index([("model_id", ASCENDING)])
            await prediction_collection.create_index([("prediction_date", DESCENDING)])
            await prediction_collection.create_index([("confidence_score", ASCENDING)])

            # Data quality logs indexes
            quality_collection = self._database[self.collections["data_quality_logs"]]
            await quality_collection.create_index([("data_source", ASCENDING)])
            await quality_collection.create_index([("quality_score", ASCENDING)])
            await quality_collection.create_index([("validation_date", DESCENDING)])

            # System metrics indexes
            metrics_collection = self._database[self.collections["system_metrics"]]
            await metrics_collection.create_index([("metric_type", ASCENDING)])
            await metrics_collection.create_index([("timestamp", DESCENDING)])

            # Text search indexes
            await crop_collection.create_index([("crop_description", TEXT)])
            await field_collection.create_index([("field_description", TEXT)])

            logger.info("MongoDB indexes setup completed")

        except Exception as e:
            logger.error("Error setting up MongoDB indexes: %s", e)
            raise

    def get_sync_client(self) -> MongoClient:
        """Returns a synchronous MongoDB client"""
        if not self._sync_client:
            self._sync_client = MongoClient(
                self.settings.mongodb_connection_string,
                maxPoolSize=self.settings.mongodb_max_pool_size,
            )
        return self._sync_client

    def get_sync_database(self):
        """Returns a synchronous MongoDB database"""
        return self.get_sync_client()[self.settings.mongodb_database]

    async def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        """Returns a specific MongoDB collection"""
        if not self._database:
            await self.connect()
        if collection_name not in self.collections:
            raise ValueError(f"Unknown collection: {collection_name}")
        return self._database[self.collections[collection_name]]

    async def health_check(self) -> Dict[str, Any]:
        """Checks the database health status"""
        try:
            if not self._database:
                await self.connect()

            start_time = time.monotonic()
            await self._client.admin.command("ping")
            ping_time = (time.monotonic() - start_time) * 1000

            db_stats = await self._database.command("dbStats")

            collection_stats = {}
            for name, coll_name in self.collections.items():
                try:
                    count = await self._database[coll_name].count_documents({})
                    collection_stats[name] = count
                except Exception:
                    collection_stats[name] = 0

            return {
                "status": "healthy",
                "ping_time_ms": round(ping_time, 2),
                "database_name": self.settings.mongodb_database,
                "database_size_mb": round(db_stats.get("dataSize", 0) / 1024 / 1024, 2),
                "collections_count": len(self.collections),
                "document_counts": collection_stats,
                "connection_verified": self._connection_verified,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def get_backup_summary(self, collection_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Returns a summary of document counts for specified collections."""
        if not collection_names:
            collection_names = list(self.collections.keys())

        backup_info: Dict[str, Any] = {
            "backup_id": str(ObjectId()),
            "timestamp": datetime.now(timezone.utc),
            "collections": {},
        }

        try:
            for coll_name in collection_names:
                if coll_name in self.collections:
                    collection = await self.get_collection(coll_name)
                    count = await collection.count_documents({})
                    backup_info["collections"][coll_name] = {
                        "document_count": count,
                        "collection_name": self.collections[coll_name],
                    }

            backup_info["status"] = "completed"
            logger.info("Backup created: %s", backup_info["backup_id"])

        except Exception as e:
            backup_info["status"] = "failed"
            backup_info["error"] = str(e)
            logger.error("Backup failed: %s", e)

        return backup_info


# Module-level singleton
_database_config_instance: Optional[DatabaseConfig] = None


def get_database_config() -> DatabaseConfig:
    """Returns a singleton DatabaseConfig instance"""
    global _database_config_instance
    if _database_config_instance is None:
        _database_config_instance = DatabaseConfig()
    return _database_config_instance


async def get_database() -> AsyncIOMotorDatabase:
    """Utility function to get a database connection"""
    db_config = get_database_config()
    return await db_config.connect()


@asynccontextmanager
async def get_db_transaction():
    """Context manager for MongoDB transactions.

    Requires a replica set. Standalone MongoDB does not support transactions.
    """
    db_config = get_database_config()
    mongo_client = db_config.client

    if not mongo_client:
        raise ConnectionError("Database client not initialized")

    try:
        async with await mongo_client.start_session() as session:
            async with session.start_transaction():
                yield session
    except ConnectionFailure as e:
        raise ConnectionError(
            "Transactions require a MongoDB replica set. "
            "Standalone MongoDB does not support multi-document transactions."
        ) from e


class MongoDBUtils:
    """Utilities for common MongoDB operations"""

    @staticmethod
    def validate_object_id(oid: str) -> ObjectId:
        try:
            return ObjectId(oid)
        except InvalidId:
            raise ValueError(f"Invalid ObjectId: {oid}")

    @staticmethod
    def serialize_document(doc: Dict[str, Any]) -> Dict[str, Any]:
        if not doc:
            return doc

        serialized = {}
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                serialized[key] = str(value)
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, dict):
                serialized[key] = MongoDBUtils.serialize_document(value)
            elif isinstance(value, list):
                serialized[key] = [
                    MongoDBUtils.serialize_document(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                serialized[key] = value
        return serialized

    @staticmethod
    def add_metadata(doc: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        metadata: Dict[str, Any] = {
            "created_at": now,
            "updated_at": now,
            "version": 1,
            "schema_version": "2.0.0",
        }
        if user_id:
            metadata["created_by"] = user_id
            metadata["updated_by"] = user_id
        return {**doc, **metadata}

    @staticmethod
    def update_metadata(doc: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        updates: Dict[str, Any] = {
            "updated_at": now,
            "version": doc.get("version", 1) + 1,
        }
        if user_id:
            updates["updated_by"] = user_id
        return updates

    @staticmethod
    def build_spatial_query(lat: float, lon: float, radius_km: float = 10) -> Dict[str, Any]:
        return {
            "coordinates": {
                "$near": {
                    "$geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    },
                    "$maxDistance": radius_km * 1000,
                }
            }
        }

    @staticmethod
    def build_date_range_query(
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        date_field: str = "date",
    ) -> Dict[str, Any]:
        query: Dict[str, Any] = {}
        if start_date or end_date:
            date_query: Dict[str, Any] = {}
            if start_date:
                date_query["$gte"] = start_date
            if end_date:
                date_query["$lte"] = end_date
            query[date_field] = date_query
        return query

    @staticmethod
    def build_aggregation_pipeline(
        match_stage: Optional[Dict[str, Any]] = None,
        group_stage: Optional[Dict[str, Any]] = None,
        sort_stage: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        pipeline: List[Dict[str, Any]] = []
        if match_stage:
            pipeline.append({"$match": match_stage})
        if group_stage:
            pipeline.append({"$group": group_stage})
        if sort_stage:
            pipeline.append({"$sort": sort_stage})
        if limit:
            pipeline.append({"$limit": limit})
        return pipeline
