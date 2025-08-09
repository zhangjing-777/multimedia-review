"""
审核结果相关数据模型
定义审核结果的数据结构和数据库映射
"""

from sqlalchemy import Column, String, Text, DateTime, JSON, Float, Integer, Boolean, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.database import Base


class ViolationType(str, enum.Enum):
    """违规类型枚举"""
    PORNOGRAPHY = "涉黄"      # 色情内容
    POLITICS = "涉政"         # 政治敏感
    VIOLENCE = "暴力"         # 暴力内容
    ADVERTISEMENT = "广告"     # 广告内容
    PROHIBITED_WORDS = "违禁词" # 违禁词汇
    TERRORISM = "恐怖主义"     # 恐怖主义
    GAMBLING = "赌博"         # 赌博内容
    DRUGS = "毒品"           # 毒品相关
    CUSTOM = "自定义"         # 自定义违规


class SourceType(str, enum.Enum):
    """识别来源类型枚举"""
    OCR = "ocr"           # OCR文字识别
    VISUAL = "visual"     # 视觉内容识别
    AUDIO = "audio"       # 音频内容识别（扩展用）
    METADATA = "metadata" # 元数据分析


class ReviewResult(Base):
    """审核结果模型"""
    
    __tablename__ = "review_results"
    
    # 主键ID
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="结果唯一标识符"
    )
    
    # 关联文件
    file_id = Column(
        UUID(as_uuid=True),
        ForeignKey("review_files.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属文件ID"
    )
    
    # 违规信息
    violation_type = Column(
        SQLEnum(ViolationType),
        nullable=False,
        comment="违规类型"
    )
    
    source_type = Column(
        SQLEnum(SourceType),
        nullable=False,
        comment="识别来源"
    )
    
    confidence_score = Column(
        Float,
        nullable=False,
        default=0.0,
        comment="置信度分数(0-1)"
    )
    
    # 违规内容详情
    evidence = Column(
        Text,
        nullable=True,
        comment="违规证据/内容描述"
    )
    
    evidence_text = Column(
        Text,
        nullable=True,
        comment="违规文本内容"
    )
    
    # 位置信息 - 支持多种坐标格式
    position = Column(
        JSON,
        nullable=True,
        comment="位置信息，格式：{'page': 1, 'bbox': [x1,y1,x2,y2]} 或 {'timestamp': 120}"
    )
    
    # 页面/帧信息
    page_number = Column(
        Integer,
        nullable=True,
        comment="页码（文档）或帧序号（视频）"
    )
    
    timestamp = Column(
        Float,
        nullable=True,
        comment="时间戳（视频，单位：秒）"
    )
    
    # AI模型信息
    model_name = Column(
        String(100),
        nullable=True,
        comment="使用的AI模型名称"
    )
    
    model_version = Column(
        String(50),
        nullable=True,
        comment="模型版本"
    )
    
    # 处理结果
    raw_response = Column(
        JSON,
        nullable=True,
        comment="AI模型原始返回结果"
    )
    
    # 人工标注
    is_reviewed = Column(
        Boolean,
        default=False,
        comment="是否已人工复审"
    )
    
    reviewer_id = Column(
        String(100),
        nullable=True,
        comment="复审人员ID"
    )
    
    review_result = Column(
        String(20),
        nullable=True,
        comment="人工复审结果：confirmed/rejected/modified"
    )
    
    review_comment = Column(
        Text,
        nullable=True,
        comment="复审备注"
    )
    
    review_time = Column(
        DateTime,
        nullable=True,
        comment="复审时间"
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
    
    # 关联关系
    file = relationship(
        "ReviewFile",
        back_populates="results",
        # comment="所属文件"
    )
    
    def __repr__(self):
        return f"<ReviewResult(id={self.id}, type='{self.violation_type}', confidence={self.confidence_score})>"
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": str(self.id),
            "file_id": str(self.file_id),
            "violation_type": self.violation_type.value,
            "source_type": self.source_type.value,
            "confidence_score": self.confidence_score,
            "evidence": self.evidence,
            "evidence_text": self.evidence_text,
            "position": self.position,
            "page_number": self.page_number,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "raw_response": self.raw_response,
            "is_reviewed": self.is_reviewed,
            "reviewer_id": self.reviewer_id,
            "review_result": self.review_result,
            "review_comment": self.review_comment,
            "review_time": self.review_time.isoformat() if self.review_time else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    @property
    def is_high_confidence(self):
        """是否高置信度（> 0.8）"""
        return self.confidence_score > 0.8
    
    @property
    def needs_review(self):
        """是否需要人工复审"""
        # 低置信度或未复审的结果需要人工处理
        return not self.is_reviewed and (
            self.confidence_score < 0.6 or 
            self.violation_type in [ViolationType.POLITICS, ViolationType.TERRORISM]
        )
    
    def mark_reviewed(self, reviewer_id: str, result: str, comment: str = None):
        """标记为已复审"""
        self.is_reviewed = True
        self.reviewer_id = reviewer_id
        self.review_result = result
        self.review_comment = comment
        self.review_time = datetime.utcnow()