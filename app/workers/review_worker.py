"""
审核任务处理器
负责异步处理审核任务和文件
"""

import asyncio
from typing import List, Dict
from celery import current_task
from sqlalchemy.orm import Session
from loguru import logger
from app.workers.celery_app import celery_app
from app.database import SessionLocal
from app.services import (
    TaskService, FileService, OCRService, 
    AIReviewService, QueueService
)
from app.models.task import TaskStatus
from app.models.file import FileStatus, FileType
from app.models.result import ReviewResult, ViolationType, SourceType
from app.utils.file_utils import FileUtils


@celery_app.task(bind=True, name="process_review_task")
def process_review_task(self, task_id: str):
    """
    处理审核任务的主流程
    
    Args:
        task_id: 任务ID
    """
    db = SessionLocal()
    try:
        task_service = TaskService(db)
        file_service = FileService(db)
        queue_service = QueueService()
        
        # 更新任务状态
        task = task_service.get_task_by_id(task_id)
        logger.info(f"开始处理任务: {task.name}")
        
        # 获取任务的所有文件
        files = task_service.get_task_files(task_id, status=FileStatus.PENDING)
        
        if not files:
            task_service.complete_task(task_id, success=False, error_message="没有待处理的文件")
            return {"status": "failed", "message": "没有待处理的文件"}
        
        # 将所有文件添加到处理队列
        for file_obj in files:
            queue_service.add_file_to_queue(
                file_id=str(file_obj.id),
                task_id=task_id,
                file_type=file_obj.file_type.value
            )
        
        # 更新任务进度
        queue_service.update_progress(
            task_id, 
            progress=10, 
            message=f"已将{len(files)}个文件加入处理队列"
        )
        
        return {
            "status": "processing",
            "message": f"任务已启动，{len(files)}个文件进入处理队列"
        }
    
    except Exception as e:
        logger.info(f"处理任务失败 {task_id}: {e}")
        
        try:
            task_service.complete_task(task_id, success=False, error_message=str(e))
        except:
            pass
        
        return {"status": "failed", "error": str(e)}
    
    finally:
        db.close()


@celery_app.task(bind=True, name="process_review_file") 
def process_review_file(self, file_id: str, task_id: str, file_type: str):
    """
    处理单个文件的审核
    
    Args:
        file_id: 文件ID
        task_id: 任务ID
        file_type: 文件类型
    """
    db = SessionLocal()
    try:
        file_service = FileService(db)
        queue_service = QueueService()
        
        # 获取文件信息
        file_obj = file_service.get_file_by_id(file_id)
        logger.info(f"开始处理文件: {file_obj.original_name}")
        
        # 更新文件状态
        file_service.update_file_status(file_id, FileStatus.PROCESSING, progress=0)
        
        # 根据文件类型进行不同处理
        if file_obj.file_type == FileType.DOCUMENT:
            result = _process_document_file(file_obj, db)
        elif file_obj.file_type == FileType.IMAGE:
            result = _process_image_file(file_obj, db)
        elif file_obj.file_type == FileType.VIDEO:
            result = _process_video_file(file_obj, db)
        elif file_obj.file_type == FileType.TEXT:
            result = _process_text_file(file_obj, db)
        else:
            raise ValueError(f"不支持的文件类型: {file_obj.file_type}")
        
        # 更新文件处理完成状态
        file_service.update_file_status(file_id, FileStatus.COMPLETED, progress=100)
        
        # 更新文件统计信息
        file_service.update_file_violation_count(file_id)
        
        # 更新任务进度
        _update_task_progress(task_id, db)
        
        logger.info(f"文件处理完成: {file_obj.original_name}, 发现{len(result)}个违规项")
        
        return {
            "status": "completed",
            "file_id": file_id,
            "violations_count": len(result),
            "message": "文件处理完成"
        }
    
    except Exception as e:
        logger.info(f"处理文件失败 {file_id}: {e}")
        
        try:
            file_service.update_file_status(
                file_id, 
                FileStatus.FAILED, 
                error_message=str(e)
            )
            _update_task_progress(task_id, db)
        except:
            pass
        
        return {"status": "failed", "file_id": file_id, "error": str(e)}
    
    finally:
        db.close()


def _process_document_file(file_obj, db: Session) -> List[Dict]:
    """处理文档文件"""
    ocr_service = OCRService()
    ai_service = AIReviewService()
    
    # 获取任务策略
    task = db.query(file_obj.task).first()
    strategy_type = task.strategy_type
    strategy_contents = task.strategy_contents
    
    # 使用同步方式运行异步函数
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # OCR提取内容
        ocr_results = loop.run_until_complete(
            ocr_service.extract_from_document(file_obj.file_path)
        )
        
        all_violations = []
        
        for page_result in ocr_results:
            page_num = page_result.get("page_number", 1)
            
            # 处理文本块
            text_blocks = [block for block in page_result.get("blocks", []) 
                          if block["type"] == "text"]
            if text_blocks:
                text_content = " ".join([block["text"] for block in text_blocks])
                text_violations = loop.run_until_complete(
                    ai_service.review_text_content(text_content, strategy_type, strategy_contents)  # 传入新参数
                )
                
                # 保存文本违规结果
                for violation in text_violations:
                    _save_violation_result(violation, file_obj.id, page_num, db)
                    all_violations.append(violation)
            
            # 处理图像块
            image_blocks = [block for block in page_result.get("blocks", []) 
                           if block["type"] == "image" and block.get("image_path")]
            for image_block in image_blocks:
                image_violations = loop.run_until_complete(
                    ai_service.review_visual_content(
                        image_block["image_path"], 
                        strategy_type, 
                        strategy_contents  # 传入新参数
                    )
                )
                
                # 保存图像违规结果
                for violation in image_violations:
                    violation["position"] = {
                        "page": page_num,
                        "bbox": image_block.get("bbox", [])
                    }
                    _save_violation_result(violation, file_obj.id, page_num, db)
                    all_violations.append(violation)
        
        # 更新OCR统计
        from app.services.file_service import FileService
        file_service = FileService(db)
        total_blocks = sum([len(r.get("blocks", [])) for r in ocr_results])
        text_blocks = sum([len([b for b in r.get("blocks", []) if b["type"] == "text"]) 
                          for r in ocr_results])
        image_blocks = total_blocks - text_blocks
        
        file_service.update_file_ocr_stats(
            str(file_obj.id), total_blocks, text_blocks, image_blocks
        )
        
        return all_violations
    
    finally:
        loop.close()


def _process_image_file(file_obj, db: Session) -> List[Dict]:
    """处理图片文件"""
    ocr_service = OCRService()
    ai_service = AIReviewService()
    
    # 获取任务策略
    task = db.query(file_obj.task).first()
    strategy_type = task.strategy_type
    strategy_contents = task.strategy_contents
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        all_violations = []
        
        # OCR提取内容
        ocr_result = loop.run_until_complete(
            ocr_service.extract_content(file_obj.file_path)
        )
        
        if ocr_result.get("success"):
            # 处理文本块
            text_blocks = [block for block in ocr_result.get("blocks", []) 
                          if block["type"] == "text"]
            if text_blocks:
                text_content = " ".join([block["text"] for block in text_blocks])
                text_violations = loop.run_until_complete(
                    ai_service.review_text_content(text_content, strategy_type, strategy_contents)  # 传入新参数
                )
                
                for violation in text_violations:
                    _save_violation_result(violation, file_obj.id, 1, db)
                    all_violations.extend(text_violations)
        
        # 直接对整个图片进行视觉审核
        visual_violations = loop.run_until_complete(
            ai_service.review_visual_content(file_obj.file_path, strategy_type, strategy_contents)  # 传入新参数
        )
        
        for violation in visual_violations:
            _save_violation_result(violation, file_obj.id, 1, db)
            all_violations.append(violation)
        
        return all_violations
    
    finally:
        loop.close()


def _process_video_file(file_obj, db: Session) -> List[Dict]:
    """处理视频文件"""
    # 获取任务配置
    task = db.query(file_obj.task).first()
    strategy_type = task.strategy_type
    strategy_contents = task.strategy_contents
    frame_interval = task.video_frame_interval
    
    # 提取视频帧
    frame_paths = FileUtils.extract_video_frames(
        file_obj.file_path, 
        interval=frame_interval,
        max_frames=100
    )
    
    if not frame_paths:
        return []
    
    ocr_service = OCRService()
    ai_service = AIReviewService()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        all_violations = []
        
        # 处理每一帧
        for i, frame_path in enumerate(frame_paths):
            frame_time = i * frame_interval  # 计算时间戳
            
            # OCR提取帧中的文本
            ocr_result = loop.run_until_complete(
                ocr_service.extract_content(frame_path)
            )
            
            if ocr_result.get("success"):
                text_blocks = [block for block in ocr_result.get("blocks", []) 
                              if block["type"] == "text"]
                if text_blocks:
                    text_content = " ".join([block["text"] for block in text_blocks])
                    text_violations = loop.run_until_complete(
                        ai_service.review_text_content(text_content, strategy_type, strategy_contents)  # 传入新参数
                    )
                    
                    for violation in text_violations:
                        violation["position"] = {"timestamp": frame_time}
                        _save_violation_result(violation, file_obj.id, None, db, frame_time)
                        all_violations.append(violation)
            
            # 视觉内容审核
            visual_violations = loop.run_until_complete(
                ai_service.review_visual_content(frame_path, strategy_type, strategy_contents)  # 传入新参数
            )
            
            for violation in visual_violations:
                violation["position"] = {"timestamp": frame_time}
                _save_violation_result(violation, file_obj.id, None, db, frame_time)
                all_violations.append(violation)
        
        # 清理临时帧文件
        FileUtils.cleanup_temp_files(frame_paths)
        
        return all_violations
    
    finally:
        loop.close()


def _process_text_file(file_obj, db: Session) -> List[Dict]:
    """处理纯文本文件"""
    ai_service = AIReviewService()
    
    # 获取任务策略
    task = db.query(file_obj.task).first()
    strategy_type = task.strategy_type
    strategy_contents = task.strategy_contents
    
    # 读取文本内容
    try:
        with open(file_obj.file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except UnicodeDecodeError:
        # 尝试其他编码
        with open(file_obj.file_path, 'r', encoding='gbk') as f:
            text_content = f.read()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # 文本审核
        violations = loop.run_until_complete(
            ai_service.review_text_content(text_content, strategy_type, strategy_contents)  # 传入新参数
        )
        
        # 保存违规结果
        for violation in violations:
            _save_violation_result(violation, file_obj.id, 1, db)
        
        return violations
    
    finally:
        loop.close()


def _save_violation_result(
    violation: Dict, 
    file_id: str, 
    page_number: int = None,
    db: Session = None,
    timestamp: float = None
):
    """保存违规结果到数据库"""
    try:
        # 转换违规类型
        violation_type = _get_violation_type_enum(violation.get("violation_type"))
        if not violation_type:
            return
        
        result = ReviewResult(
            file_id=file_id,
            violation_type=violation_type,
            source_type=SourceType(violation.get("source_type", "ocr")),
            confidence_score=violation.get("confidence_score", 0.0),
            evidence=violation.get("evidence", ""),
            evidence_text=violation.get("evidence_text"),
            position=violation.get("position"),
            page_number=page_number,
            timestamp=timestamp,
            model_name=violation.get("model_name"),
            model_version=violation.get("model_version"),
            raw_response=violation.get("raw_response")
        )
        
        db.add(result)
        db.commit()
    
    except Exception as e:
        logger.info(f"保存违规结果失败: {e}")
        db.rollback()


def _get_violation_type_enum(violation_type_str: str) -> ViolationType:
    """将字符串转换为违规类型枚举"""
    type_map = {
        "涉黄": ViolationType.PORNOGRAPHY,
        "涉政": ViolationType.POLITICS,
        "暴力": ViolationType.VIOLENCE,
        "广告": ViolationType.ADVERTISEMENT,
        "违禁词": ViolationType.PROHIBITED_WORDS,
        "恐怖主义": ViolationType.TERRORISM,
        "赌博": ViolationType.GAMBLING,
        "毒品": ViolationType.DRUGS,
    }
    return type_map.get(violation_type_str, ViolationType.CUSTOM)


def _update_task_progress(task_id: str, db: Session):
    """更新任务进度"""
    try:
        from app.services.task_service import TaskService
        from app.models.file import ReviewFile
        
        task_service = TaskService(db)
        
        # 统计已完成的文件数
        completed_files = db.query(ReviewFile).filter(
            ReviewFile.task_id == task_id,
            ReviewFile.status.in_([FileStatus.COMPLETED, FileStatus.FAILED])
        ).count()
        
        # 更新任务进度
        task_service.update_task_progress(task_id, completed_files)
        
        # 检查是否所有文件都处理完成
        total_files = db.query(ReviewFile).filter(
            ReviewFile.task_id == task_id
        ).count()
        
        if completed_files >= total_files:
            # 任务完成
            task_service.complete_task(task_id, success=True)
    
    except Exception as e:
        logger.info(f"更新任务进度失败: {e}")


@celery_app.task(name="cleanup_temp_files")
def cleanup_temp_files(file_paths: List[str]):
    """清理临时文件"""
    FileUtils.cleanup_temp_files(file_paths)
    return {"cleaned": len(file_paths)}