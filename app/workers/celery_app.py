"""
简化的 Celery 配置
去除复杂的队列配置，使用默认队列
"""

from celery import Celery
from app.config import get_settings

# 获取配置
settings = get_settings()

# 创建 Celery 实例
celery_app = Celery(
    "review_center",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.workers.review_worker"]  # 包含任务模块
)

# 简化的 Celery 配置
celery_app.conf.update(
    # 基础配置
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # 使用默认队列，不自定义路由
    task_default_queue="celery",
    task_default_exchange="celery",
    task_default_routing_key="celery",
    
    # 性能配置
    worker_prefetch_multiplier=1,  # 每次只取一个任务
    task_acks_late=True,          # 任务完成后才确认
    
    # 任务超时配置
    task_soft_time_limit=1800,    # 软超时30分钟
    task_time_limit=2100,         # 硬超时35分钟
    
    # 重试配置
    task_default_retry_delay=60,   # 默认重试延迟60秒
    task_max_retries=3,           # 最大重试3次
    
    # 结果配置
    result_expires=86400,         # 结果保存24小时
    
    # 监控配置
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # 确保任务路由到默认队列
    task_routes={
        'app.workers.review_worker.*': {'queue': 'celery'},
    },
)

# 日志配置
celery_app.conf.update(
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
)