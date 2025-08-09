"""
审核结果API路由
提供审核结果查询、人工标注等接口
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, Path, Body
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.result import ReviewResult, ViolationType, SourceType
from app.models.file import ReviewFile
from app.utils.response import APIResponse, NotFoundError

# 创建路由器
router = APIRouter()


# 请求模型定义
class MarkResultRequest(BaseModel):
    """标记审核结果请求模型"""
    reviewer_id: str = Field(..., description="复审人员ID")
    review_result: str = Field(..., pattern="^(confirmed|rejected|modified)$", description="复审结果")  # 改为 pattern
    review_comment: Optional[str] = Field(None, max_length=500, description="复审备注")


class ResultQueryParams(BaseModel):
    """结果查询参数"""
    page: int = Field(1, ge=1, description="页码")
    size: int = Field(20, ge=1, le=100, description="每页大小")
    violation_type: Optional[ViolationType] = Field(None, description="违规类型过滤")
    source_type: Optional[SourceType] = Field(None, description="来源类型过滤")
    is_reviewed: Optional[bool] = Field(None, description="是否已复审过滤")
    min_confidence: Optional[float] = Field(None, ge=0, le=1, description="最小置信度")
    file_type: Optional[str] = Field(None, description="文件类型过滤")


@router.get("/", summary="获取审核结果列表")
async def get_result_list(
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页大小"),
    task_id: Optional[str] = Query(None, description="任务ID过滤"),
    file_id: Optional[str] = Query(None, description="文件ID过滤"),
    violation_type: Optional[ViolationType] = Query(None, description="违规类型过滤"),
    source_type: Optional[SourceType] = Query(None, description="来源类型过滤"),
    is_reviewed: Optional[bool] = Query(None, description="是否已复审过滤"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1, description="最小置信度"),
    needs_review: Optional[bool] = Query(None, description="是否需要人工复审"),
    db: Session = Depends(get_db)
):
    """
    分页查询审核结果列表
    
    支持多种过滤条件：
    - **task_id**: 按任务过滤
    - **file_id**: 按文件过滤
    - **violation_type**: 按违规类型过滤
    - **source_type**: 按识别来源过滤
    - **is_reviewed**: 按是否已复审过滤
    - **min_confidence**: 按最小置信度过滤
    - **needs_review**: 按是否需要人工复审过滤
    """
    query = db.query(ReviewResult)
    
    # 文件ID过滤
    if file_id:
        query = query.filter(ReviewResult.file_id == file_id)
    
    # 任务ID过滤（通过关联文件）
    if task_id:
        query = query.join(ReviewFile).filter(ReviewFile.task_id == task_id)
    
    # 违规类型过滤
    if violation_type:
        query = query.filter(ReviewResult.violation_type == violation_type)
    
    # 来源类型过滤
    if source_type:
        query = query.filter(ReviewResult.source_type == source_type)
    
    # 是否已复审过滤
    if is_reviewed is not None:
        query = query.filter(ReviewResult.is_reviewed == is_reviewed)
    
    # 最小置信度过滤
    if min_confidence is not None:
        query = query.filter(ReviewResult.confidence_score >= min_confidence)
    
    # 是否需要人工复审过滤
    if needs_review is not None:
        if needs_review:
            # 需要复审：未复审且(低置信度或敏感类型)
            query = query.filter(
                ReviewResult.is_reviewed == False,
                db.or_(
                    ReviewResult.confidence_score < 0.6,
                    ReviewResult.violation_type.in_([
                        ViolationType.POLITICS,
                        ViolationType.TERRORISM
                    ])
                )
            )
        else:
            # 不需要复审：已复审或高置信度非敏感类型
            query = query.filter(
                db.or_(
                    ReviewResult.is_reviewed == True,
                    db.and_(
                        ReviewResult.confidence_score >= 0.6,
                        ~ReviewResult.violation_type.in_([
                            ViolationType.POLITICS,
                            ViolationType.TERRORISM
                        ])
                    )
                )
            )
    
    # 获取总数量
    total = query.count()
    
    # 分页查询
    results = query.order_by(
        ReviewResult.created_at.desc()
    ).offset((page - 1) * size).limit(size).all()
    
    # 转换为字典格式
    result_list = [result.to_dict() for result in results]
    
    return APIResponse.paginated(
        items=result_list,
        total=total,
        page=page,
        size=size,
        message="查询成功"
    )


@router.get("/{result_id}", summary="获取审核结果详情")
async def get_result_detail(
    result_id: str = Path(..., description="结果ID"),
    db: Session = Depends(get_db)
):
    """
    根据ID获取审核结果详细信息
    """
    result = db.query(ReviewResult).filter(
        ReviewResult.id == result_id
    ).first()
    
    if not result:
        raise NotFoundError(f"审核结果不存在: {result_id}")
    
    return APIResponse.success(
        data=result.to_dict(),
        message="查询成功"
    )


@router.get("/file/{file_id}", summary="获取文件的审核结果")
async def get_file_results(
    file_id: str = Path(..., description="文件ID"),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(50, ge=1, le=100, description="每页大小"),
    violation_type: Optional[ViolationType] = Query(None, description="违规类型过滤"),
    db: Session = Depends(get_db)
):
    """
    获取指定文件的所有审核结果
    """
    query = db.query(ReviewResult).filter(ReviewResult.file_id == file_id)
    
    # 违规类型过滤
    if violation_type:
        query = query.filter(ReviewResult.violation_type == violation_type)
    
    # 获取总数量
    total = query.count()
    
    # 分页查询
    results = query.order_by(
        ReviewResult.confidence_score.desc()  # 按置信度排序
    ).offset((page - 1) * size).limit(size).all()
    
    # 转换为字典格式，包含位置信息用于前端标注
    result_list = []
    for result in results:
        result_dict = result.to_dict()
        # 添加额外的前端展示信息
        result_dict["display_info"] = {
            "color": _get_violation_color(result.violation_type),
            "severity": _get_violation_severity(result.violation_type),
            "should_highlight": result.confidence_score > 0.7
        }
        result_list.append(result_dict)
    
    return APIResponse.paginated(
        items=result_list,
        total=total,
        page=page,
        size=size,
        message="文件审核结果获取成功"
    )


@router.post("/{result_id}/mark", summary="人工标记审核结果")
async def mark_result(
    request: MarkResultRequest,
    result_id: str = Path(..., description="结果ID"),
    db: Session = Depends(get_db)
):
    """
    人工标记/修改审核结果
    
    - **reviewer_id**: 复审人员ID
    - **review_result**: 复审结果 (confirmed/rejected/modified)
    - **review_comment**: 复审备注（可选）
    """
    result = db.query(ReviewResult).filter(
        ReviewResult.id == result_id
    ).first()
    
    if not result:
        raise NotFoundError(f"审核结果不存在: {result_id}")
    
    # 标记为已复审
    result.mark_reviewed(
        reviewer_id=request.reviewer_id,
        result=request.review_result,
        comment=request.review_comment
    )
    
    db.commit()
    db.refresh(result)
    
    return APIResponse.success(
        data=result.to_dict(),
        message="审核结果标记成功"
    )


@router.get("/task/{task_id}/summary", summary="获取任务审核结果摘要")
async def get_task_result_summary(
    task_id: str = Path(..., description="任务ID"),
    db: Session = Depends(get_db)
):
    """
    获取任务的审核结果摘要统计
    """
    # 违规类型统计
    violation_stats = db.query(
        ReviewResult.violation_type,
        db.func.count(ReviewResult.id).label('count'),
        db.func.avg(ReviewResult.confidence_score).label('avg_confidence')
    ).join(ReviewFile).filter(
        ReviewFile.task_id == task_id
    ).group_by(ReviewResult.violation_type).all()
    
    # 来源类型统计
    source_stats = db.query(
        ReviewResult.source_type,
        db.func.count(ReviewResult.id).label('count')
    ).join(ReviewFile).filter(
        ReviewFile.task_id == task_id
    ).group_by(ReviewResult.source_type).all()
    
    # 人工复审统计
    review_stats = db.query(
        ReviewResult.is_reviewed,
        db.func.count(ReviewResult.id).label('count')
    ).join(ReviewFile).filter(
        ReviewFile.task_id == task_id
    ).group_by(ReviewResult.is_reviewed).all()
    
    # 需要复审的结果
    need_review_count = db.query(ReviewResult).join(ReviewFile).filter(
        ReviewFile.task_id == task_id,
        ReviewResult.is_reviewed == False,
        db.or_(
            ReviewResult.confidence_score < 0.6,
            ReviewResult.violation_type.in_([
                ViolationType.POLITICS,
                ViolationType.TERRORISM
            ])
        )
    ).count()
    
    # 总违规数量
    total_violations = db.query(ReviewResult).join(ReviewFile).filter(
        ReviewFile.task_id == task_id
    ).count()
    
    summary = {
        "total_violations": total_violations,
        "need_review_count": need_review_count,
        "violation_type_stats": {
            stat.violation_type.value: {
                "count": stat.count,
                "avg_confidence": float(stat.avg_confidence) if stat.avg_confidence else 0
            }
            for stat in violation_stats
        },
        "source_type_stats": {
            stat.source_type.value: stat.count
            for stat in source_stats
        },
        "review_stats": {
            "reviewed": sum([stat.count for stat in review_stats if stat.is_reviewed]),
            "unreviewed": sum([stat.count for stat in review_stats if not stat.is_reviewed])
        }
    }
    
    return APIResponse.success(
        data=summary,
        message="任务审核结果摘要获取成功"
    )


@router.get("/pending-review", summary="获取待复审结果列表")
async def get_pending_review_results(
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页大小"),
    violation_type: Optional[ViolationType] = Query(None, description="违规类型过滤"),
    priority: str = Query("confidence", regex="^(confidence|time|type)$", description="排序优先级"),
    db: Session = Depends(get_db)
):
    """
    获取需要人工复审的结果列表
    
    - **priority**: 排序优先级 (confidence: 按置信度, time: 按时间, type: 按类型)
    """
    query = db.query(ReviewResult).filter(
        ReviewResult.is_reviewed == False,
        db.or_(
            ReviewResult.confidence_score < 0.6,
            ReviewResult.violation_type.in_([
                ViolationType.POLITICS,
                ViolationType.TERRORISM
            ])
        )
    )
    
    # 违规类型过滤
    if violation_type:
        query = query.filter(ReviewResult.violation_type == violation_type)
    
    # 排序
    if priority == "confidence":
        query = query.order_by(ReviewResult.confidence_score.asc())  # 低置信度优先
    elif priority == "time":
        query = query.order_by(ReviewResult.created_at.asc())  # 时间早的优先
    else:  # type
        query = query.order_by(ReviewResult.violation_type, ReviewResult.confidence_score.asc())
    
    # 获取总数量
    total = query.count()
    
    # 分页查询
    results = query.offset((page - 1) * size).limit(size).all()
    
    # 转换为字典格式
    result_list = [result.to_dict() for result in results]
    
    return APIResponse.paginated(
        items=result_list,
        total=total,
        page=page,
        size=size,
        message="待复审结果获取成功"
    )


def _get_violation_color(violation_type: ViolationType) -> str:
    """获取违规类型对应的颜色"""
    color_map = {
        ViolationType.PORNOGRAPHY: "#ff4757",  # 红色
        ViolationType.POLITICS: "#ff6b35",     # 橙红色
        ViolationType.VIOLENCE: "#c44569",     # 深红色
        ViolationType.ADVERTISEMENT: "#3742fa", # 蓝色
        ViolationType.PROHIBITED_WORDS: "#2ed573", # 绿色
        ViolationType.TERRORISM: "#ff3838",    # 深红色
        ViolationType.GAMBLING: "#ffa502",     # 橙色
        ViolationType.DRUGS: "#5352ed",        # 紫色
        ViolationType.CUSTOM: "#747d8c"        # 灰色
    }
    return color_map.get(violation_type, "#747d8c")


def _get_violation_severity(violation_type: ViolationType) -> str:
    """获取违规类型的严重程度"""
    severity_map = {
        ViolationType.PORNOGRAPHY: "high",
        ViolationType.POLITICS: "critical", 
        ViolationType.VIOLENCE: "high",
        ViolationType.ADVERTISEMENT: "medium",
        ViolationType.PROHIBITED_WORDS: "medium",
        ViolationType.TERRORISM: "critical",
        ViolationType.GAMBLING: "high", 
        ViolationType.DRUGS: "high",
        ViolationType.CUSTOM: "medium"
    }
    return severity_map.get(violation_type, "medium")