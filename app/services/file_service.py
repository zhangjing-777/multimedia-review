"""
文件管理服务
负责文件上传、存储、信息管理等业务逻辑
"""

from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from datetime import datetime
import os
from loguru import logger
from app.models.file import ReviewFile, FileType, FileStatus
from app.models.task import ReviewTask
from app.utils.file_utils import FileUtils
from app.utils.response import NotFoundError, BusinessError, ValidationError
from app.config import get_settings


class FileService:
    """文件管理服务类"""
    
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
    
    def upload_file(
        self,
        task_id: str,
        file_content: bytes,
        original_name: str,
        file_type: Optional[FileType] = None
    ) -> ReviewFile:
        """
        上传文件到指定任务
        
        Args:
            task_id: 所属任务ID
            file_content: 文件内容
            original_name: 原始文件名
            file_type: 文件类型（可自动检测）
            
        Returns:
            创建的文件对象
        """
        # 验证任务存在
        task = self.db.query(ReviewTask).filter(
            ReviewTask.id == task_id
        ).first()
        if not task:
            raise NotFoundError(f"任务不存在: {task_id}")
        
        # 检查任务状态
        if task.status.value == "processing":
            raise BusinessError("正在处理中的任务不能上传新文件")
        
        # 自动检测文件类型
        if not file_type:
            file_type = FileUtils.get_file_type(original_name)
        
        # 验证文件大小
        if len(file_content) > self.settings.MAX_FILE_SIZE:
            raise ValidationError(
                f"文件大小超过限制 ({len(file_content)//1024//1024}MB > "
                f"{self.settings.MAX_FILE_SIZE//1024//1024}MB)"
            )
        
        # 保存文件到磁盘
        try:
            file_path = FileUtils.save_uploaded_file(
                file_content, original_name, file_type
            )
        except Exception as e:
            raise BusinessError(f"文件保存失败: {str(e)}")
        
        # 验证保存的文件
        is_valid, error_msg = FileUtils.validate_file(file_path)
        if not is_valid:
            # 删除无效文件
            if os.path.exists(file_path):
                os.remove(file_path)
            raise ValidationError(error_msg)
        
        # 获取文件详细信息
        file_info = FileUtils.get_file_info(file_path)
        content_hash = FileUtils.get_file_hash(file_path)
        
        # 检查文件是否已存在（基于hash去重）
        existing_file = self.db.query(ReviewFile).filter(
            ReviewFile.content_hash == content_hash
        ).first()
        
        if existing_file:
            # 删除重复文件
            os.remove(file_path)
            raise BusinessError(f"文件已存在: {existing_file.original_name}")
        
        # 创建文件记录
        review_file = ReviewFile(
            task_id=task_id,
            original_name=original_name,
            file_path=file_path,
            file_type=file_type,
            file_size=len(file_content),
            mime_type=file_info.get("mime_type"),
            file_extension=file_info.get("extension"),
            content_hash=content_hash,
            page_count=file_info.get("page_count"),
            duration=file_info.get("duration"),
            status=FileStatus.PENDING
        )
        
        # 保存到数据库
        self.db.add(review_file)
        self.db.commit()
        self.db.refresh(review_file)
        
        return review_file
    
    def batch_upload_files(
        self,
        task_id: str,
        files_data: List[Dict]
    ) -> List[ReviewFile]:
        """
        批量上传文件
        
        Args:
            task_id: 任务ID
            files_data: 文件数据列表，格式：[{
                "content": bytes,
                "name": str,
                "type": FileType (可选)
            }]
            
        Returns:
            创建的文件对象列表
        """
        uploaded_files = []
        failed_files = []
        
        for file_data in files_data:
            try:
                file_obj = self.upload_file(
                    task_id=task_id,
                    file_content=file_data["content"],
                    original_name=file_data["name"],
                    file_type=file_data.get("type")
                )
                uploaded_files.append(file_obj)
            except Exception as e:
                failed_files.append({
                    "name": file_data["name"],
                    "error": str(e)
                })
        
        return uploaded_files, failed_files
    
    def get_file_by_id(self, file_id: str) -> ReviewFile:
        """
        根据ID获取文件
        
        Args:
            file_id: 文件ID
            
        Returns:
            文件对象
        """
        file_obj = self.db.query(ReviewFile).filter(
            ReviewFile.id == file_id
        ).first()
        
        if not file_obj:
            raise NotFoundError(f"文件不存在: {file_id}")
        
        return file_obj
    
    def get_files_by_task(
        self,
        task_id: str,
        status: Optional[FileStatus] = None,
        file_type: Optional[FileType] = None,
        page: int = 1,
        size: int = 50
    ) -> Tuple[List[ReviewFile], int]:
        """
        获取任务的文件列表
        
        Args:
            task_id: 任务ID
            status: 状态过滤
            file_type: 类型过滤
            page: 页码
            size: 每页大小
            
        Returns:
            (文件列表, 总数量)
        """
        query = self.db.query(ReviewFile).filter(ReviewFile.task_id == task_id)
        
        # 状态过滤
        if status:
            query = query.filter(ReviewFile.status == status)
        
        # 类型过滤
        if file_type:
            query = query.filter(ReviewFile.file_type == file_type)
        
        # 获取总数
        total = query.count()
        
        # 分页查询
        files = query.order_by(
            ReviewFile.created_at.desc()
        ).offset((page - 1) * size).limit(size).all()
        
        return files, total
    
    def update_file_status(
        self,
        file_id: str,
        status: FileStatus,
        progress: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> ReviewFile:
        """
        更新文件处理状态
        
        Args:
            file_id: 文件ID
            status: 新状态
            progress: 处理进度
            error_message: 错误信息
            
        Returns:
            更新后的文件对象
        """
        file_obj = self.get_file_by_id(file_id)
        
        file_obj.status = status
        
        if progress is not None:
            file_obj.progress = max(0, min(100, progress))
        
        if error_message is not None:
            file_obj.error_message = error_message
        
        if status == FileStatus.COMPLETED:
            file_obj.processed_at = datetime.utcnow()
            file_obj.progress = 100
        
        file_obj.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(file_obj)
        
        return file_obj
    
    def update_file_ocr_stats(
        self,
        file_id: str,
        ocr_blocks_count: int,
        text_blocks_count: int,
        image_blocks_count: int
    ) -> ReviewFile:
        """
        更新文件OCR统计信息
        
        Args:
            file_id: 文件ID
            ocr_blocks_count: OCR块总数
            text_blocks_count: 文本块数量
            image_blocks_count: 图像块数量
            
        Returns:
            更新后的文件对象
        """
        file_obj = self.get_file_by_id(file_id)
        
        file_obj.ocr_blocks_count = ocr_blocks_count
        file_obj.text_blocks_count = text_blocks_count
        file_obj.image_blocks_count = image_blocks_count
        file_obj.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(file_obj)
        
        return file_obj
    
    def update_file_violation_count(self, file_id: str) -> ReviewFile:
        """
        更新文件违规数量统计
        
        Args:
            file_id: 文件ID
            
        Returns:
            更新后的文件对象
        """
        from app.models.result import ReviewResult
        
        file_obj = self.get_file_by_id(file_id)
        
        # 计算违规数量
        violation_count = self.db.query(ReviewResult).filter(
            ReviewResult.file_id == file_id
        ).count()
        
        file_obj.violation_count = violation_count
        file_obj.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(file_obj)
        
        return file_obj
    
    def delete_file(self, file_id: str) -> bool:
        """
        删除文件（包括磁盘文件和数据库记录）
        
        Args:
            file_id: 文件ID
            
        Returns:
            是否删除成功
        """
        file_obj = self.get_file_by_id(file_id)
        
        # 检查文件状态
        if file_obj.status == FileStatus.PROCESSING:
            raise BusinessError("正在处理中的文件不能删除")
        
        # 删除磁盘文件
        try:
            if os.path.exists(file_obj.file_path):
                os.remove(file_obj.file_path)
        except Exception as e:
            logger.info(f"删除磁盘文件失败: {e}")
        
        # 删除数据库记录（会级联删除审核结果）
        self.db.delete(file_obj)
        self.db.commit()
        
        return True
    
    def get_file_content(self, file_id: str) -> bytes:
        """
        获取文件内容
        
        Args:
            file_id: 文件ID
            
        Returns:
            文件内容字节
        """
        file_obj = self.get_file_by_id(file_id)
        
        if not os.path.exists(file_obj.file_path):
            raise NotFoundError("文件不存在于磁盘")
        
        try:
            with open(file_obj.file_path, "rb") as f:
                return f.read()
        except Exception as e:
            raise BusinessError(f"读取文件失败: {str(e)}")
    
    def get_files_by_status(
        self,
        status: FileStatus,
        limit: int = 100
    ) -> List[ReviewFile]:
        """
        根据状态获取文件列表
        
        Args:
            status: 文件状态
            limit: 返回数量限制
            
        Returns:
            文件列表
        """
        return self.db.query(ReviewFile).filter(
            ReviewFile.status == status
        ).order_by(
            ReviewFile.created_at.asc()
        ).limit(limit).all()
    
    def cleanup_failed_uploads(self, hours: int = 24) -> int:
        """
        清理失败的上传文件
        
        Args:
            hours: 清理多少小时前的失败文件
            
        Returns:
            清理的文件数量
        """
        from datetime import timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # 查找需要清理的文件
        failed_files = self.db.query(ReviewFile).filter(
            ReviewFile.status == FileStatus.FAILED,
            ReviewFile.created_at < cutoff_time
        ).all()
        
        cleaned_count = 0
        for file_obj in failed_files:
            try:
                # 删除磁盘文件
                if os.path.exists(file_obj.file_path):
                    os.remove(file_obj.file_path)
                
                # 删除数据库记录
                self.db.delete(file_obj)
                cleaned_count += 1
            except Exception as e:
                logger.info(f"清理失败文件失败 {file_obj.id}: {e}")
        
        if cleaned_count > 0:
            self.db.commit()
        
        return cleaned_count