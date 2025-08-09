"""
多媒体审核任务中心
主应用包初始化文件
"""

__version__ = "1.0.0"
__author__ = "Review Team"
__description__ = "多媒体内容审核任务中心"

# 导入核心组件，方便外部使用
from .config import get_settings
from .database import get_db, init_database

# 导出常用类型
from .models import (
    ReviewTask, ReviewFile, ReviewResult,
    TaskStatus, FileStatus, ViolationType
)

# 包级别的配置
__all__ = [
    "get_settings",
    "get_db", 
    "init_database",
    "ReviewTask",
    "ReviewFile", 
    "ReviewResult",
    "TaskStatus",
    "FileStatus",
    "ViolationType",
]