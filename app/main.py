"""
FastAPI 主应用入口
配置应用、路由、中间件等
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger
import sys

from app.config import get_settings, ensure_upload_dir
from app.database import init_database, health_check
from app.api import task, upload, result
from app.utils.response import APIResponse


# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭时的生命周期管理"""
    
    # 启动时执行
    logger.info("🚀 应用启动中...")
    
    try:
        # 确保上传目录存在
        ensure_upload_dir()
        logger.info("✅ 上传目录初始化完成")
        
        # 初始化数据库
        init_database()
        logger.info("✅ 数据库初始化完成")
        
        # 健康检查
        health_status = health_check()
        if health_status["status"] == "healthy":
            logger.info("✅ 依赖服务连接正常")
        else:
            logger.warning(f"⚠️ 部分依赖服务异常: {health_status}")
        
        logger.info("🎉 应用启动完成")
        
    except Exception as e:
        logger.error(f"❌ 应用启动失败: {e}")
        sys.exit(1)
    
    yield  # 应用运行中
    
    # 关闭时执行
    logger.info("👋 应用关闭中...")
    logger.info("✅ 应用关闭完成")


# 创建FastAPI应用实例
def create_app() -> FastAPI:
    """创建并配置FastAPI应用"""
    
    settings = get_settings()
    
    # 创建应用实例
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="多媒体内容审核任务中心 - 支持文档/图片/视频的AI智能审核",
        docs_url="/docs" if settings.DEBUG else None,  # 生产环境可关闭文档
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan
    )
    
    # 配置CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应该限制具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 配置日志
    configure_logging(settings)
    
    # 注册路由
    register_routes(app)
    
    # 注册异常处理器
    register_exception_handlers(app)
    
    return app


def configure_logging(settings):
    """配置日志系统"""
    
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True
    )
    
    # 添加文件输出
    if settings.LOG_FILE:
        logger.add(
            settings.LOG_FILE,
            level=settings.LOG_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="100MB",  # 日志轮转
            retention="30 days",  # 保留30天
            compression="zip"  # 压缩旧日志
        )


def register_routes(app: FastAPI):
    """注册所有路由"""
    
    # API路由前缀
    API_V1_PREFIX = "/api/v1"
    
    # 注册各模块路由
    app.include_router(
        task.router,
        prefix=f"{API_V1_PREFIX}/tasks",
        tags=["任务管理"]
    )
    
    app.include_router(
        upload.router,
        prefix=f"{API_V1_PREFIX}/upload",
        tags=["文件上传"]
    )
    
    app.include_router(
        result.router,
        prefix=f"{API_V1_PREFIX}/results",
        tags=["审核结果"]
    )
    
    # 健康检查端点
    @app.get("/health", tags=["系统"])
    async def health_check_endpoint():
        """健康检查接口"""
        health_status = health_check()
        return APIResponse.success(
            data=health_status,
            message="健康检查完成"
        )
    
    # 根路径
    @app.get("/", tags=["系统"])
    async def root():
        """根路径欢迎信息"""
        settings = get_settings()
        return APIResponse.success(
            data={
                "name": settings.APP_NAME,
                "version": settings.VERSION,
                "docs_url": "/docs",
                "health_url": "/health"
            },
            message="欢迎使用多媒体审核任务中心"
        )


def register_exception_handlers(app: FastAPI):
    """注册全局异常处理器"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP异常处理"""
        logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail}")
        
        # 如果detail已经是标准格式，直接返回
        if isinstance(exc.detail, dict) and "success" in exc.detail:
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.detail
            )
        
        # 否则包装为标准格式
        return JSONResponse(
            status_code=exc.status_code,
            content=APIResponse.error(
                message=str(exc.detail),
                code=exc.status_code
            )
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """通用异常处理"""
        logger.error(f"未处理异常: {type(exc).__name__}: {str(exc)}")
        
        # 开发环境返回详细错误信息
        settings = get_settings()
        if settings.DEBUG:
            import traceback
            error_detail = traceback.format_exc()
        else:
            error_detail = "服务器内部错误"
        
        return JSONResponse(
            status_code=500,
            content=APIResponse.error(
                message=error_detail,
                code=500,
                error_type="InternalServerError"
            )
        )
    
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        """请求日志中间件"""
        import time
        
        start_time = time.time()
        
        # 记录请求开始
        logger.info(f"📥 {request.method} {request.url}")
        
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录响应
        logger.info(
            f"📤 {request.method} {request.url} - "
            f"状态码: {response.status_code} - "
            f"耗时: {process_time:.3f}s"
        )
        
        # 添加处理时间到响应头
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


# 创建应用实例
app = create_app()


# 开发服务器启动
if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )