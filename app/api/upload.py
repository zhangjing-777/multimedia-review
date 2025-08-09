"""
文件上传API路由
提供文件上传、状态查询等接口
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, File, UploadFile, Form, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.file_service import FileService
from app.services.queue_service import QueueService
from app.models.file import FileType, FileStatus
from app.utils.response import APIResponse, ValidationError
from app.config import get_settings

# 创建路由器
router = APIRouter()


# 响应模型定义
class UploadResponse(BaseModel):
    """上传响应模型"""
    file_id: str
    original_name: str
    file_type: str
    file_size: int
    status: str
    message: str


class BatchUploadResponse(BaseModel):
    """批量上传响应模型"""
    success_count: int
    failed_count: int
    success_files: List[UploadResponse]
    failed_files: List[dict]


@router.post("/single", summary="单文件上传")
async def upload_single_file(
    task_id: str = Form(..., description="任务ID"),
    file: UploadFile = File(..., description="上传的文件"),
    db: Session = Depends(get_db)
):
    """
    上传单个文件到指定任务
    
    - **task_id**: 目标任务ID
    - **file**: 要上传的文件
    
    支持的文件类型：
    - 文档：PDF, DOCX, DOC, TXT
    - 图片：JPG, JPEG, PNG, GIF, BMP
    - 视频：MP4, AVI, MOV, WMV, FLV
    """
    settings = get_settings()
    file_service = FileService(db)
    queue_service = QueueService()
    
    # 验证文件大小
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise ValidationError(
            f"文件大小超过限制 ({file.size//1024//1024}MB > "
            f"{settings.MAX_FILE_SIZE//1024//1024}MB)"
        )
    
    # 验证文件类型
    if file.filename:
        from pathlib import Path
        ext = Path(file.filename).suffix.lower().lstrip('.')
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise ValidationError(f"不支持的文件类型: {ext}")
    
    try:
        # 读取文件内容
        file_content = await file.read()
        
        # 上传文件
        uploaded_file = file_service.upload_file(
            task_id=task_id,
            file_content=file_content,
            original_name=file.filename or "unknown"
        )
        
        # 添加到处理队列
        queue_service.add_file_to_queue(
            file_id=str(uploaded_file.id),
            task_id=task_id,
            file_type=uploaded_file.file_type.value
        )
        
        return APIResponse.success(
            data=UploadResponse(
                file_id=str(uploaded_file.id),
                original_name=uploaded_file.original_name,
                file_type=uploaded_file.file_type.value,
                file_size=uploaded_file.file_size,
                status=uploaded_file.status.value,
                message="文件上传成功"
            ),
            message="文件上传成功"
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件上传失败: {str(e)}")


@router.post("/batch", summary="批量文件上传")
async def upload_batch_files(
    task_id: str = Form(..., description="任务ID"),
    files: List[UploadFile] = File(..., description="上传的文件列表"),
    db: Session = Depends(get_db)
):
    """
    批量上传文件到指定任务
    
    - **task_id**: 目标任务ID
    - **files**: 要上传的文件列表（最多20个）
    """
    settings = get_settings()
    
    # 限制批量上传数量
    if len(files) > 20:
        raise ValidationError("批量上传最多支持20个文件")
    
    file_service = FileService(db)
    queue_service = QueueService()
    
    success_files = []
    failed_files = []
    
    for file in files:
        try:
            # 验证单个文件
            if file.size and file.size > settings.MAX_FILE_SIZE:
                failed_files.append({
                    "name": file.filename,
                    "error": f"文件大小超过限制 ({file.size//1024//1024}MB)"
                })
                continue
            
            if file.filename:
                from pathlib import Path
                ext = Path(file.filename).suffix.lower().lstrip('.')
                if ext not in settings.ALLOWED_EXTENSIONS:
                    failed_files.append({
                        "name": file.filename,
                        "error": f"不支持的文件类型: {ext}"
                    })
                    continue
            
            # 读取文件内容
            file_content = await file.read()
            
            # 上传文件
            uploaded_file = file_service.upload_file(
                task_id=task_id,
                file_content=file_content,
                original_name=file.filename or "unknown"
            )
            
            # 添加到处理队列
            queue_service.add_file_to_queue(
                file_id=str(uploaded_file.id),
                task_id=task_id,
                file_type=uploaded_file.file_type.value
            )
            
            success_files.append(UploadResponse(
                file_id=str(uploaded_file.id),
                original_name=uploaded_file.original_name,
                file_type=uploaded_file.file_type.value,
                file_size=uploaded_file.file_size,
                status=uploaded_file.status.value,
                message="上传成功"
            ))
        
        except Exception as e:
            failed_files.append({
                "name": file.filename,
                "error": str(e)
            })
    
    return APIResponse.success(
        data=BatchUploadResponse(
            success_count=len(success_files),
            failed_count=len(failed_files),
            success_files=success_files,
            failed_files=failed_files
        ),
        message=f"批量上传完成，成功{len(success_files)}个，失败{len(failed_files)}个"
    )


@router.get("/status/{file_id}", summary="查询文件处理状态")
async def get_file_status(
    file_id: str,
    db: Session = Depends(get_db)
):
    """
    查询文件处理状态和进度
    
    - **file_id**: 文件ID
    """
    file_service = FileService(db)
    queue_service = QueueService()
    
    # 获取文件信息
    file_obj = file_service.get_file_by_id(file_id)
    
    # 获取队列中的状态
    queue_status = queue_service.get_file_status(file_id)
    queue_progress = queue_service.get_progress(file_id)
    
    status_data = {
        "file_info": file_obj.to_dict(),
        "queue_status": queue_status,
        "queue_progress": queue_progress
    }
    
    return APIResponse.success(
        data=status_data,
        message="状态查询成功"
    )


@router.get("/status", summary="批量查询文件状态")
async def get_batch_file_status(
    file_ids: str = Query(..., description="文件ID列表，逗号分隔"),
    db: Session = Depends(get_db)
):
    """
    批量查询多个文件的处理状态
    
    - **file_ids**: 文件ID列表，使用逗号分隔，如：id1,id2,id3
    """
    file_service = FileService(db)
    queue_service = QueueService()
    
    # 解析文件ID列表
    id_list = [fid.strip() for fid in file_ids.split(',') if fid.strip()]
    
    if len(id_list) > 50:
        raise ValidationError("批量查询最多支持50个文件")
    
    status_list = []
    
    for file_id in id_list:
        try:
            file_obj = file_service.get_file_by_id(file_id)
            queue_status = queue_service.get_file_status(file_id)
            queue_progress = queue_service.get_progress(file_id)
            
            status_list.append({
                "file_id": file_id,
                "file_info": file_obj.to_dict(),
                "queue_status": queue_status,
                "queue_progress": queue_progress
            })
        
        except Exception as e:
            status_list.append({
                "file_id": file_id,
                "error": str(e)
            })
    
    return APIResponse.success(
        data=status_list,
        message="批量状态查询完成"
    )


@router.delete("/{file_id}", summary="删除文件")
async def delete_file(
    file_id: str,
    db: Session = Depends(get_db)
):
    """
    删除指定文件
    
    注意：正在处理中的文件不能删除
    """
    file_service = FileService(db)
    
    success = file_service.delete_file(file_id)
    
    return APIResponse.success(
        data={"deleted": success},
        message="文件删除成功"
    )


@router.get("/{file_id}/download", summary="下载文件")
async def download_file(
    file_id: str,
    db: Session = Depends(get_db)
):
    """
    下载原始文件
    """
    from fastapi.responses import Response
    
    file_service = FileService(db)
    
    # 获取文件信息
    file_obj = file_service.get_file_by_id(file_id)
    
    # 读取文件内容
    file_content = file_service.get_file_content(file_id)
    
    # 返回文件响应
    return Response(
        content=file_content,
        media_type=file_obj.mime_type or "application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename={file_obj.original_name}"
        }
    )


@router.get("/queue/status", summary="获取上传队列状态")
async def get_queue_status():
    """
    获取文件处理队列状态
    
    返回队列中等待处理的文件数量等信息
    """
    queue_service = QueueService()
    
    status = queue_service.get_queue_status()
    
    return APIResponse.success(
        data=status,
        message="队列状态获取成功"
    )