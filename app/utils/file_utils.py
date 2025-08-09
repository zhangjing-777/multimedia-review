"""
文件处理工具模块
提供文件上传、类型检测、格式转换等功能
"""

import os
import hashlib
import shutil
import mimetypes
from typing import Optional, Tuple, List
from pathlib import Path
import cv2
import uuid
from PIL import Image
from PyPDF2 import PdfReader
from docx import Document
from loguru import logger
from app.config import get_settings
from app.models.file import FileType


class FileUtils:
    """文件处理工具类"""
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """
        计算文件MD5哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            MD5哈希值
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                # 分块读取避免大文件占用过多内存
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.info(f"计算文件哈希失败: {e}")
            return ""
    
    @staticmethod
    def get_file_type(filename: str) -> FileType:
        """
        根据文件扩展名判断文件类型
        
        Args:
            filename: 文件名
            
        Returns:
            文件类型枚举
        """
        ext = Path(filename).suffix.lower().lstrip('.')
        
        # 文档类型
        if ext in ['pdf', 'doc', 'docx', 'txt', 'rtf']:
            return FileType.DOCUMENT
        
        # 图片类型
        elif ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
            return FileType.IMAGE
        
        # 视频类型
        elif ext in ['mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv']:
            return FileType.VIDEO
        
        # 默认为文本
        else:
            return FileType.TEXT
    
    @staticmethod
    def validate_file(file_path: str, max_size: int = None) -> Tuple[bool, str]:
        """
        验证文件是否有效
        
        Args:
            file_path: 文件路径
            max_size: 最大文件大小（字节）
            
        Returns:
            (是否有效, 错误信息)
        """
        settings = get_settings()
        max_size = max_size or settings.MAX_FILE_SIZE
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return False, "文件不存在"
        
        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size > max_size:
            return False, f"文件大小超过限制 ({file_size // (1024*1024)}MB > {max_size // (1024*1024)}MB)"
        
        # 检查文件扩展名
        filename = os.path.basename(file_path)
        ext = Path(filename).suffix.lower().lstrip('.')
        if ext not in settings.allowed_extensions_set:
            return False, f"不支持的文件类型: {ext}"
        
        return True, ""
    
    @staticmethod
    def save_uploaded_file(file_content: bytes, original_name: str, file_type: FileType) -> str:
        """
        保存上传的文件
        
        Args:
            file_content: 文件内容
            original_name: 原始文件名
            file_type: 文件类型
            
        Returns:
            保存的文件路径
        """
        settings = get_settings()
        
        # 确定子目录
        type_dir_map = {
            FileType.DOCUMENT: "documents",
            FileType.IMAGE: "images", 
            FileType.VIDEO: "videos",
            FileType.TEXT: "documents"
        }
        
        subdir = type_dir_map.get(file_type, "others")
        save_dir = os.path.join(settings.UPLOAD_DIR, subdir)
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成唯一文件名
        ext = Path(original_name).suffix
        unique_name = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(save_dir, unique_name)
        
        # 保存文件
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return file_path
    
    @staticmethod
    def get_file_info(file_path: str) -> dict:
        """
        获取文件详细信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件信息字典
        """
        if not os.path.exists(file_path):
            return {}
        
        stat = os.stat(file_path)
        filename = os.path.basename(file_path)
        
        info = {
            "size": stat.st_size,
            "mime_type": mimetypes.guess_type(file_path)[0],
            "extension": Path(filename).suffix.lower().lstrip('.'),
            "created_time": stat.st_ctime,
            "modified_time": stat.st_mtime,
        }
        
        # 根据文件类型获取额外信息
        file_type = FileUtils.get_file_type(filename)
        
        if file_type == FileType.DOCUMENT:
            info.update(FileUtils._get_document_info(file_path))
        elif file_type == FileType.IMAGE:
            info.update(FileUtils._get_image_info(file_path))
        elif file_type == FileType.VIDEO:
            info.update(FileUtils._get_video_info(file_path))
        
        return info
    
    @staticmethod
    def _get_document_info(file_path: str) -> dict:
        """获取文档信息"""
        info = {"page_count": 0}
        
        try:
            ext = Path(file_path).suffix.lower()
            
            if ext == '.pdf':
                # PDF文档
                with open(file_path, 'rb') as f:
                    reader = PdfReader(f)
                    info["page_count"] = len(reader.pages)
            
            elif ext in ['.docx', '.doc']:
                # Word文档
                doc = Document(file_path)
                info["page_count"] = len(doc.element.body)  # 简化的页数计算
            
            elif ext == '.txt':
                # 文本文件按行数估算页数
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = len(f.readlines())
                    info["page_count"] = max(1, lines // 50)  # 假设每页50行
        
        except Exception as e:
            logger.info(f"获取文档信息失败: {e}")
        
        return info
    
    @staticmethod
    def _get_image_info(file_path: str) -> dict:
        """获取图片信息"""
        info = {"width": 0, "height": 0}
        
        try:
            with Image.open(file_path) as img:
                info["width"] = img.width
                info["height"] = img.height
                info["format"] = img.format
                info["mode"] = img.mode
        
        except Exception as e:
            logger.info(f"获取图片信息失败: {e}")
        
        return info
    
    @staticmethod
    def _get_video_info(file_path: str) -> dict:
        """获取视频信息"""
        info = {"duration": 0, "frame_count": 0, "fps": 0}
        
        try:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                info.update({
                    "duration": int(duration),
                    "frame_count": frame_count,
                    "fps": fps,
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                })
            
            cap.release()
        
        except Exception as e:
            logger.info(f"获取视频信息失败: {e}")
        
        return info
    
    @staticmethod
    def extract_video_frames(
        video_path: str, 
        interval: int = 5, 
        max_frames: int = 100
    ) -> List[str]:
        """
        从视频中提取关键帧
        
        Args:
            video_path: 视频文件路径
            interval: 抽帧间隔（秒）
            max_frames: 最大帧数
            
        Returns:
            帧图片路径列表
        """
        frame_paths = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return frame_paths
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * interval)  # 间隔帧数
            
            # 创建临时目录保存帧
            temp_dir = os.path.join(
                get_settings().UPLOAD_DIR, 
                "temp", 
                f"frames_{uuid.uuid4()}"
            )
            os.makedirs(temp_dir, exist_ok=True)
            
            frame_num = 0
            saved_frames = 0
            
            while saved_frames < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按间隔保存帧
                if frame_num % frame_interval == 0:
                    frame_path = os.path.join(temp_dir, f"frame_{frame_num:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    saved_frames += 1
                
                frame_num += 1
            
            cap.release()
        
        except Exception as e:
            logger.info(f"视频抽帧失败: {e}")
        
        return frame_paths
    
    @staticmethod
    def cleanup_temp_files(file_paths: List[str]):
        """清理临时文件"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    # 如果是目录，递归删除
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    else:
                        os.remove(file_path)
            except Exception as e:
                logger.info(f"清理临时文件失败 {file_path}: {e}")