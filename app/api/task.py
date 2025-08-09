"""
任务管理API路由
提供任务创建、查询、更新、删除等接口
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, Path
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from loguru import logger
from app.database import get_db
from app.services.task_service import TaskService
from app.services.queue_service import QueueService
from app.models.task import TaskStatus, StrategyType
from app.utils.response import APIResponse, paginated_response

# 创建路由器
router = APIRouter()


# 请求模型定义
class CreateTaskRequest(BaseModel):
    """创建任务请求模型"""
    name: str = Field(..., min_length=1, max_length=200, description="任务名称")
    description: Optional[str] = Field(None, max_length=1000, description="任务描述")
    strategy_type: Optional[str] = Field(None, max_length=100, description="审核策略类型")  # 改为字符串
    strategy_contents: Optional[str] = Field(None, description="审核策略内容")  # 改为字符串
    video_frame_interval: int = Field(5, ge=1, le=60, description="视频抽帧间隔(秒)")
    creator_id: Optional[str] = Field(None, description="创建者ID")


class UpdateTaskRequest(BaseModel):
    """更新任务请求模型"""
    name: Optional[str] = Field(None, min_length=1, max_length=200, description="任务名称")
    description: Optional[str] = Field(None, max_length=1000, description="任务描述")
    strategy_type: Optional[str] = Field(None, max_length=100, description="审核策略类型")  # 新增
    strategy_contents: Optional[str] = Field(None, description="审核策略内容")  # 改为字符串
    video_frame_interval: Optional[int] = Field(None, ge=1, le=60, description="视频抽帧间隔")


class TaskQueryParams(BaseModel):
    """任务查询参数"""
    page: int = Field(1, ge=1, description="页码")
    size: int = Field(20, ge=1, le=100, description="每页大小")
    status: Optional[TaskStatus] = Field(None, description="状态过滤")
    strategy_type: Optional[str] = Field(None, description="策略类型过滤")  # 改为字符串
    creator_id: Optional[str] = Field(None, description="创建者过滤")
    keyword: Optional[str] = Field(None, description="关键词搜索")


# API接口实现
@router.post("/", summary="创建审核任务")
async def create_task(
    request: CreateTaskRequest,
    db: Session = Depends(get_db)
):
    """
    创建新的审核任务
    
    - **name**: 任务名称，必填
    - **description**: 任务描述，可选
    - **strategy_type**: 审核策略类型
    - **strategy_contents**: 具体的审核策略内容
    - **video_frame_interval**: 视频抽帧间隔，默认5秒
    - **creator_id**: 创建者ID，可选
    """
    task_service = TaskService(db)
    
    task = task_service.create_task(
        name=request.name,
        description=request.description,
        strategy_type=request.strategy_type,
        strategy_contents=request.strategy_contents,
        video_frame_interval=request.video_frame_interval,
        creator_id=request.creator_id
    )
    
    return APIResponse.success(
        data=task.to_dict(),
        message="任务创建成功"
    )


@router.get("/", summary="获取任务列表")
async def get_task_list(
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页大小"),
    status: Optional[TaskStatus] = Query(None, description="状态过滤"),
    strategy_type: Optional[str] = Query(None, description="策略类型过滤"),  
    creator_id: Optional[str] = Query(None, description="创建者过滤"),
    keyword: Optional[str] = Query(None, description="关键词搜索"),
    db: Session = Depends(get_db)
):
    """
    分页查询任务列表
    
    支持多种过滤条件：
    - 状态过滤
    - 策略类型过滤
    - 创建者过滤
    - 关键词搜索（任务名称和描述）
    """
    task_service = TaskService(db)
    
    tasks, total = task_service.get_task_list(
        page=page,
        size=size,
        status=status,
        strategy_type=strategy_type,
        creator_id=creator_id,
        keyword=keyword
    )
    
    # 转换为字典格式
    task_list = [task.to_dict() for task in tasks]
    
    return APIResponse.paginated(
        items=task_list,
        total=total,
        page=page,
        size=size,
        message="查询成功"
    )


@router.get("/{task_id}", summary="获取任务详情")
async def get_task_detail(
    task_id: str = Path(..., description="任务ID"),
    db: Session = Depends(get_db)
):
    """
    根据ID获取任务详细信息
    """
    task_service = TaskService(db)
    task = task_service.get_task_by_id(task_id)
    
    return APIResponse.success(
        data=task.to_dict(),
        message="查询成功"
    )


@router.put("/{task_id}", summary="更新任务信息")
async def update_task(
    request: UpdateTaskRequest,
    task_id: str = Path(..., description="任务ID"),
    db: Session = Depends(get_db)
):
    """
    更新任务信息
    
    只能更新未开始处理的任务
    """
    task_service = TaskService(db)
    
    task = task_service.update_task(
        task_id=task_id,
        name=request.name,
        description=request.description,
        strategy_type=request.strategy_type,
        strategy_contents=request.strategy_contents,
        video_frame_interval=request.video_frame_interval
    )
    
    return APIResponse.success(
        data=task.to_dict(),
        message="任务更新成功"
    )


@router.delete("/{task_id}", summary="删除任务")
async def delete_task(
    task_id: str = Path(..., description="任务ID"),
    db: Session = Depends(get_db)
):
    """
    删除任务及其所有关联数据
    
    注意：正在处理中的任务不能删除
    """
    task_service = TaskService(db)
    
    success = task_service.delete_task(task_id)
    
    return APIResponse.success(
        data={"deleted": success},
        message="任务删除成功"
    )


@router.post("/{task_id}/start", summary="启动任务处理")
async def start_task(
    task_id: str = Path(..., description="任务ID"),
    db: Session = Depends(get_db)
):
    """启动任务处理"""
    
    task_service = TaskService(db)
    queue_service = QueueService()
    
    # 启动任务
    success = task_service.start_task(task_id)
    logger.info(f"任务启动结果: {success}")
    
    if success:
        # 添加到处理队列
        queue_result = queue_service.add_task_to_queue(task_id)
        logger.info(f"队列添加结果: {queue_result}")
        
        # 检查队列状态
        queue_status = queue_service.get_queue_status()
        logger.info(f"队列状态: {queue_status}")
    
    return APIResponse.success(
        data={"started": success},
        message="任务启动成功"
    )


@router.post("/{task_id}/cancel", summary="取消任务处理")
async def cancel_task(
    task_id: str = Path(..., description="任务ID"),
    db: Session = Depends(get_db)
):
    """
    取消任务处理
    
    只能取消等待中或处理中的任务
    """
    task_service = TaskService(db)
    
    task = task_service.cancel_task(task_id)
    
    return APIResponse.success(
        data=task.to_dict(),
        message="任务取消成功"
    )


@router.post("/{task_id}/recheck", summary="重新审核任务")
async def recheck_task(
    task_id: str = Path(..., description="任务ID"),
    db: Session = Depends(get_db)
):
    """
    重新审核任务
    
    重置任务状态，清除旧的审核结果，重新开始处理
    """
    task_service = TaskService(db)
    queue_service = QueueService()
    
    # 重置任务
    success = task_service.recheck_task(task_id)
    
    if success:
        # 添加到处理队列
        queue_service.add_task_to_queue(task_id)
    
    return APIResponse.success(
        data={"rechecked": success},
        message="任务重新审核启动成功"
    )


@router.get("/{task_id}/statistics", summary="获取任务统计")
async def get_task_statistics(
    task_id: str = Path(..., description="任务ID"),
    db: Session = Depends(get_db)
):
    """
    获取任务详细统计信息
    
    包括文件处理状态、违规类型统计、处理时长等
    """
    task_service = TaskService(db)
    
    stats = task_service.get_task_statistics(task_id)
    
    return APIResponse.success(
        data=stats,
        message="统计信息获取成功"
    )


@router.get("/{task_id}/progress", summary="获取任务处理进度")
async def get_task_progress(
    task_id: str = Path(..., description="任务ID"),
    db: Session = Depends(get_db)
):
    """
    获取任务实时处理进度
    
    返回任务的处理进度和状态信息
    """
    task_service = TaskService(db)
    queue_service = QueueService()
    
    # 获取任务基本信息
    task = task_service.get_task_by_id(task_id)
    
    # 获取队列中的进度信息
    queue_progress = queue_service.get_progress(task_id)
    
    progress_data = {
        "task_id": task_id,
        "status": task.status.value,
        "progress": task.progress,
        "total_files": task.total_files,
        "processed_files": task.processed_files,
        "violation_count": task.violation_count,
        "queue_progress": queue_progress
    }
    
    return APIResponse.success(
        data=progress_data,
        message="进度获取成功"
    )


@router.get("/{task_id}/files", summary="获取任务文件列表")
async def get_task_files(
    task_id: str = Path(..., description="任务ID"),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页大小"),
    db: Session = Depends(get_db)
):
    """
    获取任务关联的文件列表
    """
    from app.services.file_service import FileService
    
    file_service = FileService(db)
    
    files, total = file_service.get_files_by_task(
        task_id=task_id,
        page=page,
        size=size
    )
    
    file_list = [file.to_dict() for file in files]
    
    return APIResponse.paginated(
        items=file_list,
        total=total,
        page=page,
        size=size,
        message="文件列表获取成功"
    )