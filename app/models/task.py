"""
任务相关数据模型
定义审核任务的数据结构和数据库映射
"""

from sqlalchemy import Column, String, Text, DateTime, JSON, Enum as SQLEnum, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.database import Base


class TaskStatus(str, enum.Enum):
    """任务状态枚举"""
    PENDING = "pending"        # 等待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"         # 处理失败
    CANCELLED = "cancelled"   # 已取消


class StrategyType(str, enum.Enum):
    """审核策略类型枚举"""
    CONTENT_SAFETY = "content_safety"    # 内容安全审核
    ADVERTISEMENT = "advertisement"      # 广告识别
    COPYRIGHT = "copyright"             # 版权检测
    CUSTOM = "custom"                   # 自定义策略


class ReviewTask(Base):
    """审核任务模型"""
    
    __tablename__ = "review_tasks"
    
    # 主键ID
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="任务唯一标识符"
    )
    
    # 任务基本信息
    name = Column(
        String(200),
        nullable=False,
        comment="任务名称"
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="任务描述"
    )
    
    # 审核策略配置
    strategy_type = Column(
        String(100),
        nullable=True,
        comment="审核策略类型"
    )

    strategy_contents = Column(
        Text,
        nullable=True,
        comment="审核策略内容"
    )
    
    # 处理配置
    video_frame_interval = Column(
        Integer,
        default=5,
        comment="视频抽帧间隔(秒)"
    )
    
    # 任务状态
    status = Column(
        SQLEnum(TaskStatus),
        nullable=False,
        default=TaskStatus.PENDING,
        comment="任务状态"
    )
    
    progress = Column(
        Integer,
        default=0,
        comment="处理进度(0-100)"
    )
    
    error_message = Column(
        Text,
        nullable=True,
        comment="错误信息"
    )
    
    # 统计信息
    total_files = Column(
        Integer,
        default=0,
        comment="总文件数"
    )
    
    processed_files = Column(
        Integer,
        default=0,
        comment="已处理文件数"
    )
    
    violation_count = Column(
        Integer,
        default=0,
        comment="违规文件数量"
    )
    
    # 创建者信息（可扩展用户系统）
    creator_id = Column(
        String(100),
        nullable=True,
        comment="创建者ID"
    )
    
    # 时间戳
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        comment="创建时间"
    )
    
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="更新时间"
    )
    
    started_at = Column(
        DateTime,
        nullable=True,
        comment="开始处理时间"
    )
    
    completed_at = Column(
        DateTime,
        nullable=True,
        comment="完成时间"
    )
    
    # 关联关系
    files = relationship(
        "ReviewFile",
        back_populates="task",
        cascade="all, delete-orphan",
        # comment="关联的文件列表"
    )
    
    def __repr__(self):
        return f"<ReviewTask(id={self.id}, name='{self.name}', status='{self.status}')>"
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "strategy_type": self.strategy_type,
            "strategy_contents": self.strategy_contents,
            "video_frame_interval": self.video_frame_interval,
            "status": self.status.value,
            "progress": self.progress,
            "error_message": self.error_message,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "violation_count": self.violation_count,
            "creator_id": self.creator_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
    
    def update_progress(self):
        """更新任务进度"""
        if self.total_files > 0:
            self.progress = int((self.processed_files / self.total_files) * 100)
        else:
            self.progress = 0