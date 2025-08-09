"""
重新设计的消息队列服务
直接与 Celery 集成，简化队列管理
"""

from typing import Dict, Any, Optional, List
import json
from datetime import datetime
from loguru import logger

from app.config import get_settings


class QueueService:
    """简化的消息队列服务类，直接使用 Celery"""
    
    def __init__(self):
        self.settings = get_settings()
        
        # 直接使用 Celery 的 Redis 连接
        import redis
        self.redis = redis.Redis.from_url(
            self.settings.CELERY_BROKER_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        
        # 状态存储的 Redis 连接（使用缓存数据库）
        self.cache_redis = redis.Redis.from_url(
            self.settings.REDIS_CACHE_URL,
            decode_responses=True
        )
        
        logger.info(f"QueueService 初始化完成，使用 Celery Broker: {self.settings.CELERY_BROKER_URL}")
    
    def add_task_to_queue(self, task_id: str, priority: int = 0) -> bool:
        """
        将任务添加到 Celery 队列（直接调用 Celery 任务）
        
        Args:
            task_id: 任务ID
            priority: 优先级（暂时不使用）
            
        Returns:
            是否添加成功
        """
        try:
            # 直接调用 Celery 任务，不走 Redis 队列
            from app.workers.review_worker import process_review_task
            
            logger.info(f"🚀 提交任务到 Celery: {task_id}")
            result = process_review_task.delay(task_id)
            
            # 保存 Celery 任务ID到缓存
            self.set_task_status(task_id, "submitted", {
                "celery_task_id": result.id,
                "priority": priority
            })
            
            logger.info(f"✅ 任务已提交，Celery任务ID: {result.id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 提交任务失败: {e}")
            return False
    
    def add_file_to_queue(self, file_id: str, task_id: str, file_type: str, priority: int = 0) -> bool:
        """
        将文件添加到处理队列（直接调用 Celery 任务）
        
        Args:
            file_id: 文件ID
            task_id: 所属任务ID
            file_type: 文件类型
            priority: 优先级
            
        Returns:
            是否添加成功
        """
        try:
            from app.workers.review_worker import process_review_file
            
            logger.info(f"🚀 提交文件处理任务: {file_id}")
            result = process_review_file.delay(file_id, task_id, file_type)
            
            # 保存状态
            self.set_file_status(file_id, "submitted", {
                "celery_task_id": result.id,
                "task_id": task_id,
                "file_type": file_type
            })
            
            logger.info(f"✅ 文件任务已提交，Celery任务ID: {result.id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 提交文件任务失败: {e}")
            return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        try:
            key = f"task_status:{task_id}"
            status_json = self.cache_redis.get(key)
            
            if status_json:
                return json.loads(status_json)
            return None
            
        except Exception as e:
            logger.error(f"获取任务状态失败: {e}")
            return None
    
    def set_task_status(self, task_id: str, status: str, extra_data: Dict = None) -> bool:
        """设置任务状态"""
        try:
            status_data = {
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if extra_data:
                status_data.update(extra_data)
            
            key = f"task_status:{task_id}"
            self.cache_redis.setex(key, 86400, json.dumps(status_data))  # 24小时过期
            return True
            
        except Exception as e:
            logger.error(f"设置任务状态失败: {e}")
            return False
    
    def get_file_status(self, file_id: str) -> Optional[Dict]:
        """获取文件状态"""
        try:
            key = f"file_status:{file_id}"
            status_json = self.cache_redis.get(key)
            
            if status_json:
                return json.loads(status_json)
            return None
            
        except Exception as e:
            logger.error(f"获取文件状态失败: {e}")
            return None
    
    def set_file_status(self, file_id: str, status: str, extra_data: Dict = None) -> bool:
        """设置文件状态"""
        try:
            status_data = {
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if extra_data:
                status_data.update(extra_data)
            
            key = f"file_status:{file_id}"
            self.cache_redis.setex(key, 86400, json.dumps(status_data))
            return True
            
        except Exception as e:
            logger.error(f"设置文件状态失败: {e}")
            return False
    
    def update_progress(self, entity_id: str, progress: int, message: str = "") -> bool:
        """更新处理进度"""
        try:
            progress_data = {
                "progress": max(0, min(100, progress)),
                "message": message,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            key = f"progress:{entity_id}"
            self.cache_redis.setex(key, 3600, json.dumps(progress_data))  # 1小时过期
            return True
            
        except Exception as e:
            logger.error(f"更新进度失败: {e}")
            return False
    
    def get_progress(self, entity_id: str) -> Optional[Dict]:
        """获取处理进度"""
        try:
            key = f"progress:{entity_id}"
            progress_json = self.cache_redis.get(key)
            
            if progress_json:
                return json.loads(progress_json)
            return None
            
        except Exception as e:
            logger.error(f"获取进度失败: {e}")
            return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        获取队列状态统计
        基于 Celery 的统计信息
        """
        try:
            from celery import current_app
            
            # 获取 Celery 状态
            inspect = current_app.control.inspect()
            
            # 获取活动任务
            active_tasks = inspect.active()
            reserved_tasks = inspect.reserved()
            
            # 统计信息
            total_active = sum(len(tasks) for tasks in (active_tasks or {}).values())
            total_reserved = sum(len(tasks) for tasks in (reserved_tasks or {}).values())
            
            return {
                "active_tasks": total_active,
                "reserved_tasks": total_reserved,
                "total_pending": total_active + total_reserved,
                "workers_online": len(active_tasks or {}),
                "queue_info": "使用 Celery 原生队列管理"
            }
            
        except Exception as e:
            logger.error(f"获取队列状态失败: {e}")
            return {
                "active_tasks": 0,
                "reserved_tasks": 0,
                "total_pending": 0,
                "workers_online": 0,
                "error": str(e)
            }
    
    def get_celery_task_result(self, celery_task_id: str) -> Optional[Dict]:
        """获取 Celery 任务结果"""
        try:
            from celery.result import AsyncResult
            
            result = AsyncResult(celery_task_id)
            return {
                "task_id": celery_task_id,
                "status": result.status,
                "result": result.result,
                "traceback": result.traceback,
                "successful": result.successful()
            }
            
        except Exception as e:
            logger.error(f"获取 Celery 任务结果失败: {e}")
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        try:
            # 获取 Celery 任务ID
            task_status = self.get_task_status(task_id)
            if not task_status or "celery_task_id" not in task_status:
                return False
            
            # 取消 Celery 任务
            from celery import current_app
            current_app.control.revoke(task_status["celery_task_id"], terminate=True)
            
            # 更新状态
            self.set_task_status(task_id, "cancelled")
            
            logger.info(f"✅ 任务已取消: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"取消任务失败: {e}")
            return False
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            # 测试 Celery Redis 连接
            self.redis.ping()
            
            # 测试缓存 Redis 连接
            self.cache_redis.ping()
            
            logger.info("✅ QueueService 连接测试成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ QueueService 连接测试失败: {e}")
            return False