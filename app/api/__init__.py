"""
API包初始化文件
导出所有API路由模块
"""

from . import task, upload, result

# 导出所有路由模块
__all__ = [
    "task",
    "upload", 
    "result",
]