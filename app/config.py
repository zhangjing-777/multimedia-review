"""
配置管理模块
负责管理应用的所有配置项，支持环境变量和默认值
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用基础配置
    APP_NAME: str = "多媒体审核任务中心"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # 数据库配置
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/review_center"
    DATABASE_ECHO: bool = False  # 是否打印SQL语句
    
    # Redis配置
    REDIS_URL: str = "redis://localhost:6379/1"
    REDIS_CACHE_URL: str = "redis://localhost:6379/2"  # 缓存专用数据库
    
    # Celery配置
    CELERY_BROKER_URL: str = "redis://localhost:6379/3"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/4"
    
    # 文件存储配置   
    UPLOAD_DIR: str = "./uploads"  # 上传文件存储目录
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 最大文件大小100MB
    ALLOWED_EXTENSIONS: str = "pdf,docx,doc,txt,jpg,jpeg,png,gif,mp4,avi,mov,wmv"  # 改为字符串
    # ALLOWED_EXTENSIONS: set = {
    #     "pdf", "docx", "doc", "txt",  # 文档类型
    #     "jpg", "jpeg", "png", "gif",  # 图片类型
    #     "mp4", "avi", "mov", "wmv"   # 视频类型
    # }
    
    # AI服务配置
    OCR_API_URL: str = "http://localhost:8001/ocr"  # OCR服务地址
    OPENROUTER_VISION_MODEL: str = "qwen/qwen2.5-vl-32b-instruct:free" # 视觉语言模型服务
    OPENROUTER_TEXT_MODEL: str =  "" # 大语言模型服务
    OPENROUTER_API_KEY: str = ""
    ENDPOINT: str = "https://openrouter.ai/api/v1/chat/completions"
    
    # 视频处理配置
    DEFAULT_FRAME_INTERVAL: int = 5  # 默认抽帧间隔(秒)
    MAX_FRAMES_PER_VIDEO: int = 100  # 每个视频最大抽帧数
    
    # 审核策略配置
    # DEFAULT_STRATEGIES: list = [
    #     "涉黄", "涉政", "暴力", "广告", "违禁词"
    # ]
    
    # 缓存配置
    CACHE_TTL: int = 3600  # 缓存过期时间(秒)
    RESULT_CACHE_TTL: int = 7200  # 审核结果缓存时间(秒)
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    class Config:
        """Pydantic配置"""
        env_file = ".env"
        case_sensitive = True
        env_file_encoding = 'utf-8'
    
    @property
    def max_file_size_int(self) -> int:
        return int(self.MAX_FILE_SIZE)

    @property
    def allowed_extensions_set(self) -> set:
        """将字符串转换为set"""
        return set(ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(','))


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置实例（单例模式）
    使用lru_cache确保配置只初始化一次
    """
    return Settings()


# 创建上传目录
def ensure_upload_dir():
    """确保上传目录存在"""
    settings = get_settings()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # 创建子目录
    subdirs = ["documents", "images", "videos", "temp"]
    for subdir in subdirs:
        os.makedirs(os.path.join(settings.UPLOAD_DIR, subdir), exist_ok=True)