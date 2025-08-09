"""
任务管理服务
负责审核任务的创建、查询、更新、删除等业务逻辑
"""

from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from datetime import datetime
import uuid

from app.models.task import ReviewTask, TaskStatus, StrategyType
from app.models.file import ReviewFile, FileStatus
from app.models.result import ReviewResult, ViolationType
from app.utils.response import NotFoundError, BusinessError


class TaskService:
    """任务管理服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_task(
        self,
        name: str,
        description: Optional[str] = None,
        strategy_type: Optional[str] = None,  # 改为字符串
        strategy_contents: Optional[str] = None,  # 改为字符串
        video_frame_interval: int = 5,
        creator_id: Optional[str] = None
    ) -> ReviewTask:
        """
        创建审核任务
        
        Args:
            name: 任务名称
            description: 任务描述
            strategy_type: 审核策略类型
            strategy_contents: 审核策略内容
            video_frame_interval: 视频抽帧间隔
            creator_id: 创建者ID
            
        Returns:
            创建的任务对象
        """
        # 参数验证
        if not name or not name.strip():
            raise BusinessError("任务名称不能为空")
        
        # 创建任务对象
        task = ReviewTask(
            name=name.strip(),
            description=description,
            strategy_type=strategy_type,
            strategy_contents=strategy_contents,
            video_frame_interval=max(1, video_frame_interval),  # 最小间隔1秒
            creator_id=creator_id,
            status=TaskStatus.PENDING
        )
        
        # 保存到数据库
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        
        return task
    
    def get_task_by_id(self, task_id: str) -> ReviewTask:
        """
        根据ID获取任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务对象
            
        Raises:
            NotFoundError: 任务不存在
        """
        task = self.db.query(ReviewTask).filter(
            ReviewTask.id == task_id
        ).first()
        
        if not task:
            raise NotFoundError(f"任务不存在: {task_id}")
        
        return task
    
    def get_task_list(
        self,
        page: int = 1,
        size: int = 20,
        status: Optional[TaskStatus] = None,
        strategy_type: Optional[str] = None,
        creator_id: Optional[str] = None,
        keyword: Optional[str] = None
    ) -> Tuple[List[ReviewTask], int]:
        """
        获取任务列表（分页查询）
        
        Args:
            page: 页码
            size: 每页大小
            status: 任务状态过滤
            strategy_type: 策略类型过滤
            creator_id: 创建者过滤
            keyword: 关键词搜索（任务名称和描述）
            
        Returns:
            (任务列表, 总数量)
        """
        query = self.db.query(ReviewTask)
        
        # 状态过滤
        if status:
            query = query.filter(ReviewTask.status == status)
        
        # 策略类型过滤
        if strategy_type:
            query = query.filter(ReviewTask.strategy_type == strategy_type)
        
        # 创建者过滤
        if creator_id:
            query = query.filter(ReviewTask.creator_id == creator_id)
        
        # 关键词搜索
        if keyword and keyword.strip():
            keyword = f"%{keyword.strip()}%"
            query = query.filter(
                or_(
                    ReviewTask.name.ilike(keyword),
                    ReviewTask.description.ilike(keyword)
                )
            )
        
        # 获取总数量
        total = query.count()
        
        # 分页查询
        tasks = query.order_by(
            ReviewTask.created_at.desc()
        ).offset((page - 1) * size).limit(size).all()
        
        return tasks, total
    
    def update_task(
        self,
        task_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        strategy_type: Optional[str] = None,
        strategy_contents: Optional[str] = None,
        video_frame_interval: Optional[int] = None
    ) -> ReviewTask:
        """
        更新任务信息
        
        Args:
            task_id: 任务ID
            name: 新的任务名称
            description: 新的任务描述
            strategy_contents: 新的审核策略
            video_frame_interval: 新的抽帧间隔
            
        Returns:
            更新后的任务对象
        """
        task = self.get_task_by_id(task_id)
        
        # 检查任务状态
        if task.status == TaskStatus.PROCESSING:
            raise BusinessError("正在处理中的任务不能修改")
        
        # 更新字段
        if name is not None:
            if not name.strip():
                raise BusinessError("任务名称不能为空")
            task.name = name.strip()
        
        if description is not None:
            task.description = description
        
        if strategy_type is not None:
            task.strategy_type = strategy_type

        if strategy_contents is not None:
            task.strategy_contents = strategy_contents
        
        if video_frame_interval is not None:
            task.video_frame_interval = max(1, video_frame_interval)
        
        task.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(task)
        
        return task
    
    def delete_task(self, task_id: str) -> bool:
        """
        删除任务（级联删除文件和结果）
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否删除成功
        """
        task = self.get_task_by_id(task_id)
        
        # 检查任务状态
        if task.status == TaskStatus.PROCESSING:
            raise BusinessError("正在处理中的任务不能删除")
        
        # 删除任务（数据库设置了级联删除）
        self.db.delete(task)
        self.db.commit()
        
        return True
    
    def start_task(self, task_id: str) -> bool:
        """
        启动任务处理
        """
        task = self.get_task_by_id(task_id)
        
        # 允许的启动状态
        allowed_statuses = [
            TaskStatus.PENDING,      # 等待中
            TaskStatus.CANCELLED,    # 已取消
            TaskStatus.FAILED,       # 失败
            TaskStatus.COMPLETED     # 已完成（重新处理）
        ]
        
        if task.status not in allowed_statuses:
            raise BusinessError(f"任务状态不允许启动，当前状态: {task.status.value}")
        
        # 检查是否有文件
        file_count = self.db.query(ReviewFile).filter(
            ReviewFile.task_id == task_id
        ).count()
        
        if file_count == 0:
            raise BusinessError("任务没有关联文件，无法开始处理")
        
        # 如果是重新启动，重置相关状态
        if task.status in [TaskStatus.CANCELLED, TaskStatus.FAILED, TaskStatus.COMPLETED]:
            # 重置文件状态
            self.db.query(ReviewFile).filter(
                ReviewFile.task_id == task_id
            ).update({
                "status": FileStatus.PENDING,
                "progress": 0,
                "error_message": None,
                "processed_at": None
            })
            
            # 清除旧的审核结果（可选）
            # self.db.query(ReviewResult).filter(
            #     ReviewResult.file_id.in_(
            #         self.db.query(ReviewFile.id).filter(ReviewFile.task_id == task_id)
            #     )
            # ).delete(synchronize_session=False)
        
        # 更新任务状态
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.utcnow()
        task.total_files = file_count
        task.processed_files = 0
        task.violation_count = 0
        task.progress = 0
        task.error_message = None
        task.completed_at = None
        
        self.db.commit()
        
        return True
    
    def update_task_progress(self, task_id: str, processed_files: int = None) -> ReviewTask:
        """
        更新任务处理进度
        
        Args:
            task_id: 任务ID
            processed_files: 已处理文件数
            
        Returns:
            更新后的任务对象
        """
        task = self.get_task_by_id(task_id)
        
        if processed_files is not None:
            task.processed_files = processed_files
        
        # 更新进度百分比
        task.update_progress()
        
        # 更新违规文件统计
        violation_count = self.db.query(ReviewFile).filter(
            and_(
                ReviewFile.task_id == task_id,
                ReviewFile.violation_count > 0
            )
        ).count()
        task.violation_count = violation_count
        
        task.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(task)
        
        return task
    
    def complete_task(self, task_id: str, success: bool = True, error_message: str = None) -> ReviewTask:
        """
        完成任务处理
        
        Args:
            task_id: 任务ID
            success: 是否成功完成
            error_message: 错误信息（失败时）
            
        Returns:
            更新后的任务对象
        """
        task = self.get_task_by_id(task_id)
        
        if success:
            task.status = TaskStatus.COMPLETED
        else:
            task.status = TaskStatus.FAILED
            task.error_message = error_message
        
        task.completed_at = datetime.utcnow()
        task.update_progress()
        
        # 更新最终统计
        violation_count = self.db.query(ReviewFile).filter(
            and_(
                ReviewFile.task_id == task_id,
                ReviewFile.violation_count > 0
            )
        ).count()
        task.violation_count = violation_count
        
        self.db.commit()
        self.db.refresh(task)
        
        return task
    
    def cancel_task(self, task_id: str) -> ReviewTask:
        """
        取消任务处理
        
        Args:
            task_id: 任务ID
            
        Returns:
            更新后的任务对象
        """
        task = self.get_task_by_id(task_id)
        
        # 只有等待中或处理中的任务可以取消
        if task.status not in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
            raise BusinessError(f"任务状态不允许取消，当前状态: {task.status.value}")
        
        task.status = TaskStatus.CANCELLED
        task.updated_at = datetime.utcnow()
        
        # 取消所有未完成的文件处理
        self.db.query(ReviewFile).filter(
            and_(
                ReviewFile.task_id == task_id,
                ReviewFile.status.in_([FileStatus.PENDING, FileStatus.PROCESSING])
            )
        ).update({"status": FileStatus.CANCELLED})
        
        self.db.commit()
        self.db.refresh(task)
        
        return task
    
    def get_task_statistics(self, task_id: str) -> Dict:
        """
        获取任务统计信息
        
        Args:
            task_id: 任务ID
            
        Returns:
            统计信息字典
        """
        task = self.get_task_by_id(task_id)
        
        # 文件状态统计
        file_stats = self.db.query(
            ReviewFile.status,
            func.count(ReviewFile.id).label('count')
        ).filter(
            ReviewFile.task_id == task_id
        ).group_by(ReviewFile.status).all()
        
        file_status_counts = {status.value: 0 for status in FileStatus}
        for status, count in file_stats:
            file_status_counts[status.value] = count
        
        # 违规类型统计
        violation_stats = self.db.query(
            ReviewResult.violation_type,
            func.count(ReviewResult.id).label('count')
        ).join(ReviewFile).filter(
            ReviewFile.task_id == task_id
        ).group_by(ReviewResult.violation_type).all()
        
        violation_counts = {}
        for violation_type, count in violation_stats:
            violation_counts[violation_type.value] = count
        
        # 处理时长
        processing_duration = None
        if task.started_at and task.completed_at:
            duration = task.completed_at - task.started_at
            processing_duration = int(duration.total_seconds())
        
        return {
            "task_info": task.to_dict(),
            "file_status_counts": file_status_counts,
            "violation_counts": violation_counts,
            "processing_duration": processing_duration,
            "total_violations": sum(violation_counts.values()),
            "completion_rate": task.progress
        }
    
    def recheck_task(self, task_id: str) -> bool:
        """
        重新审核任务（重置状态并重新处理）
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功启动重新审核
        """
        task = self.get_task_by_id(task_id)
        
        # 只有已完成或失败的任务可以重新审核
        if task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            raise BusinessError(f"任务状态不允许重新审核，当前状态: {task.status.value}")
        
        # 重置任务状态
        task.status = TaskStatus.PENDING
        task.progress = 0
        task.processed_files = 0
        task.violation_count = 0
        task.error_message = None
        task.started_at = None
        task.completed_at = None
        task.updated_at = datetime.utcnow()
        
        # 重置文件状态
        self.db.query(ReviewFile).filter(
            ReviewFile.task_id == task_id
        ).update({
            "status": FileStatus.PENDING,
            "progress": 0,
            "error_message": None,
            "processed_at": None
        })
        
        # 删除旧的审核结果
        self.db.query(ReviewResult).filter(
            ReviewResult.file_id.in_(
                self.db.query(ReviewFile.id).filter(
                    ReviewFile.task_id == task_id
                )
            )
        ).delete(synchronize_session=False)
        
        self.db.commit()
        
        return True
    
    def get_task_files(self, task_id: str, status: Optional[FileStatus] = None) -> List[ReviewFile]:
        """
        获取任务关联的文件列表
        
        Args:
            task_id: 任务ID
            status: 文件状态过滤
            
        Returns:
            文件列表
        """
        # 验证任务存在
        self.get_task_by_id(task_id)
        
        query = self.db.query(ReviewFile).filter(ReviewFile.task_id == task_id)
        
        if status:
            query = query.filter(ReviewFile.status == status)
        
        return query.order_by(ReviewFile.created_at.asc()).all()
    
    def get_pending_tasks(self, limit: int = 10) -> List[ReviewTask]:
        """
        获取等待处理的任务列表
        
        Args:
            limit: 返回数量限制
            
        Returns:
            等待处理的任务列表
        """
        return self.db.query(ReviewTask).filter(
            ReviewTask.status == TaskStatus.PENDING
        ).order_by(
            ReviewTask.created_at.asc()
        ).limit(limit).all()