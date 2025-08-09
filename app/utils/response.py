"""
统一响应格式工具
提供标准化的API响应格式和常用响应方法
"""

from typing import Any, Optional, Dict, List
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime


class APIResponse:
    """API响应格式化类"""
    
    @staticmethod
    def success(
        data: Any = None,
        message: str = "操作成功",
        code: int = 200
    ) -> Dict[str, Any]:
        """
        成功响应格式
        
        Args:
            data: 响应数据
            message: 响应消息
            code: 状态码
            
        Returns:
            标准化的成功响应
        """
        return {
            "success": True,
            "code": code,
            "message": message,
            "data": data,
            "timestamp": int(datetime.utcnow().timestamp())
        }
    
    @staticmethod
    def error(
        message: str = "操作失败",
        code: int = 400,
        data: Any = None,
        error_type: str = "BadRequest"
    ) -> Dict[str, Any]:
        """
        错误响应格式
        
        Args:
            message: 错误消息
            code: 状态码
            data: 错误详情数据
            error_type: 错误类型
            
        Returns:
            标准化的错误响应
        """
        return {
            "success": False,
            "code": code,
            "message": message,
            "error_type": error_type,
            "data": data,
            "timestamp": int(datetime.utcnow().timestamp())
        }
    
    @staticmethod
    def paginated(
        items: List[Any],
        total: int,
        page: int = 1,
        size: int = 20,
        message: str = "查询成功"
    ) -> Dict[str, Any]:
        """
        分页响应格式
        
        Args:
            items: 数据列表
            total: 总数量
            page: 当前页码
            size: 每页大小
            message: 响应消息
            
        Returns:
            标准化的分页响应
        """
        total_pages = (total + size - 1) // size  # 向上取整
        
        return APIResponse.success(
            data={
                "items": items,
                "pagination": {
                    "total": total,
                    "page": page,
                    "size": size,
                    "pages": total_pages,
                    "has_next": page < total_pages,
                    "has_prev": page > 1
                }
            },
            message=message
        )


# 常用HTTP异常
class APIException(HTTPException):
    """自定义API异常"""
    
    def __init__(
        self,
        message: str,
        code: int = 400,
        error_type: str = "APIError"
    ):
        super().__init__(
            status_code=code,
            detail=APIResponse.error(
                message=message,
                code=code,
                error_type=error_type
            )
        )


# 预定义常用异常
class ValidationError(APIException):
    """参数验证错误"""
    def __init__(self, message: str = "参数验证失败"):
        super().__init__(message, 400, "ValidationError")


class NotFoundError(APIException):
    """资源不存在错误"""
    def __init__(self, message: str = "资源不存在"):
        super().__init__(message, 404, "NotFoundError")


class PermissionError(APIException):
    """权限不足错误"""
    def __init__(self, message: str = "权限不足"):
        super().__init__(message, 403, "PermissionError")


class ServerError(APIException):
    """服务器内部错误"""
    def __init__(self, message: str = "服务器内部错误"):
        super().__init__(message, 500, "ServerError")


class BusinessError(APIException):
    """业务逻辑错误"""
    def __init__(self, message: str = "业务处理失败"):
        super().__init__(message, 400, "BusinessError")


# 响应工具函数
def success_response(
    data: Any = None,
    message: str = "操作成功"
) -> JSONResponse:
    """返回成功响应"""
    return JSONResponse(
        content=APIResponse.success(data=data, message=message),
        status_code=200
    )


def error_response(
    message: str = "操作失败",
    code: int = 400
) -> JSONResponse:
    """返回错误响应"""
    return JSONResponse(
        content=APIResponse.error(message=message, code=code),
        status_code=code
    )


def paginated_response(
    items: List[Any],
    total: int,
    page: int = 1,
    size: int = 20
) -> JSONResponse:
    """返回分页响应"""
    return JSONResponse(
        content=APIResponse.paginated(items, total, page, size),
        status_code=200
    )
