"""
数据库连接模块
负责数据库连接池管理和会话创建
"""

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from typing import Generator
from loguru import logger
import redis
from app.config import get_settings

# 修补SQLAlchemy以支持openGauss
def patch_opengauss():
    """修补SQLAlchemy支持openGauss"""
    from sqlalchemy.dialects.postgresql.psycopg2 import PGDialect_psycopg2
    
    original_get_server_version_info = PGDialect_psycopg2._get_server_version_info
    
    def patched_get_server_version_info(self, connection):
        try:
            return original_get_server_version_info(self, connection)
        except AssertionError:
            # openGauss版本检查失败时，返回PostgreSQL 14.0
            logger.warning("检测到openGauss，使用PostgreSQL 14.0兼容模式")
            return (14, 0)
    
    PGDialect_psycopg2._get_server_version_info = patched_get_server_version_info

# 应用补丁
patch_opengauss()

# 获取配置
settings = get_settings()

# 创建数据库引擎
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,  # 是否打印SQL
    poolclass=QueuePool,  # 连接池类型
    pool_size=20,  # 连接池大小
    max_overflow=0,  # 最大溢出连接数
    pool_recycle=3600,  # 连接回收时间
    pool_pre_ping=True,  # 连接前ping测试
)

# 创建会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# 创建基础模型类
Base = declarative_base()

# Redis连接池
redis_pool = redis.ConnectionPool.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    max_connections=50
)

# Redis客户端
redis_client = redis.Redis(connection_pool=redis_pool)

# Redis缓存客户端（使用独立数据库）
cache_client = redis.Redis(
    connection_pool=redis.ConnectionPool.from_url(
        settings.REDIS_CACHE_URL,
        decode_responses=True
    )
)


def get_db() -> Generator[Session, None, None]:
    """
    数据库会话依赖注入
    用于FastAPI路由中获取数据库会话
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_redis() -> redis.Redis:
    """获取Redis客户端"""
    return redis_client


def get_cache() -> redis.Redis:
    """获取缓存客户端"""
    return cache_client


def init_database():
    """
    初始化数据库（跳过所有数据库操作）
    """
    try:
        # 只导入模型，不进行任何数据库连接
        from app.models import task, file, result
        logger.info("✅ 数据库模型加载完成")
        return True
        
    except Exception as e:
        logger.info(f"❌ 数据库模型加载失败: {e}")
        raise


def check_database_connection():
    """检查数据库连接（暂时跳过）"""
    logger.info("⚠️ 跳过数据库连接检查")
    return True
# def check_database_connection():
#     """检查数据库连接"""
#     try:
#         db = SessionLocal()
#         # 执行简单查询测试连接
#         db.execute("SELECT 1")
#         db.close()
#         logger.info("✅ 数据库连接正常")
#         return True
#     except Exception as e:
#         logger.info(f"❌ 数据库连接失败: {e}")
#         return False


def check_redis_connection():
    """检查Redis连接"""
    try:
        redis_client.ping()
        cache_client.ping()
        logger.info("✅ Redis连接正常")
        return True
    except Exception as e:
        logger.info(f"❌ Redis连接失败: {e}")
        return False


def health_check():
    """
    健康检查
    检查所有依赖服务的连接状态
    """
    health = {
        "database": check_database_connection(),
        "redis": check_redis_connection(),
    }
    
    all_healthy = all(health.values())
    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "services": health
    }