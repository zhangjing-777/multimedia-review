"""
文件相关数据模型
定义上传文件的数据结构和数据库映射
"""

from sqlalchemy import Column, String, Text, DateTime, Integer, BigInteger, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum
import os

from app.database import Base


class FileType(str, enum.Enum):
    """文件类型枚举"""
    DOCUMENT = "document"  # 文档类型 (PDF, DOCX, TXT)
    IMAGE = "image"       # 图片类型 (JPG, PNG, GIF)
    VIDEO = "video"       # 视频类型 (MP4, AVI, MOV)
    TEXT = "text"         # 纯文本


class FileStatus(str, enum.Enum):
    """文件处理状态枚举"""
    PENDING = "pending"        # 等待处理
    UPLOADING = "uploading"    # 上传中
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"    # 处理完成
    FAILED = "failed"         # 处理失败
    CANCELLED = "cancelled"   # 已取消


class ReviewFile(Base):
    """审核文件模型"""
    
    __tablename__ = "review_files"
    
    # 主键ID
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="文件唯一标识符"
    )
    
    # 关联任务
    task_id = Column(
        UUID(as_uuid=True),
        ForeignKey("review_tasks.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属任务ID"
    )
    
    # 文件基本信息
    original_name = Column(
        String(500),
        nullable=False,
        comment="原始文件名"
    )
    
    file_path = Column(
        String(1000),
        nullable=False,
        comment="文件存储路径"
    )
    
    file_type = Column(
        SQLEnum(FileType),
        nullable=False,
        comment="文件类型"
    )
    
    file_size = Column(
        BigInteger,
        nullable=False,
        comment="文件大小(字节)"
    )
    
    mime_type = Column(
        String(100),
        nullable=True,
        comment="MIME类型"
    )
    
    file_extension = Column(
        String(10),
        nullable=True,
        comment="文件扩展名"
    )
    
    # 文件内容信息
    content_hash = Column(
        String(64),
        nullable=True,
        comment="文件内容MD5哈希值，用于去重"
    )
    
    page_count = Column(
        Integer,
        nullable=True,
        comment="页数/帧数（文档/视频）"
    )
    
    duration = Column(
        Integer,
        nullable=True,
        comment="时长(秒)，仅视频文件"
    )
    
    # 处理状态
    status = Column(
        SQLEnum(FileStatus),
        nullable=False,
        default=FileStatus.PENDING,
        comment="处理状态"
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
    
    # 处理结果统计
    ocr_blocks_count = Column(
        Integer,
        default=0,
        comment="OCR识别的内容块数量"
    )
    
    text_blocks_count = Column(
        Integer,
        default=0,
        comment="文本块数量"
    )
    
    image_blocks_count = Column(
        Integer,
        default=0,
        comment="图像块数量"
    )
    
    violation_count = Column(
        Integer,
        default=0,
        comment="违规内容数量"
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
    
    processed_at = Column(
        DateTime,
        nullable=True,
        comment="处理完成时间"
    )
    
    # 关联关系
    task = relationship(
        "ReviewTask",
        back_populates="files",
        # comment="所属任务"
    )
    
    results = relationship(
        "ReviewResult",
        back_populates="file",
        cascade="all, delete-orphan",
        # comment="审核结果列表"
    )
    
    def __repr__(self):
        return f"<ReviewFile(id={self.id}, name='{self.original_name}', status='{self.status}')>"
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": str(self.id),
            "task_id": str(self.task_id),
            "original_name": self.original_name,
            "file_path": self.file_path,
            "file_type": self.file_type.value,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "file_extension": self.file_extension,
            "content_hash": self.content_hash,
            "page_count": self.page_count,
            "duration": self.duration,
            "status": self.status.value,
            "progress": self.progress,
            "error_message": self.error_message,
            "ocr_blocks_count": self.ocr_blocks_count,
            "text_blocks_count": self.text_blocks_count,
            "image_blocks_count": self.image_blocks_count,
            "violation_count": self.violation_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }
    
    @property
    def file_size_mb(self):
        """文件大小(MB)"""
        return round(self.file_size / (1024 * 1024), 2) if self.file_size else 0
    
    @property
    def exists(self):
        """检查文件是否存在"""
        return os.path.exists(self.file_path) if self.file_path else False
    
    def get_relative_path(self):
        """获取相对路径"""
        from app.config import get_settings
        settings = get_settings()
        
        if self.file_path and self.file_path.startswith(settings.UPLOAD_DIR):
            return os.path.relpath(self.file_path, settings.UPLOAD_DIR)
        return self.file_path