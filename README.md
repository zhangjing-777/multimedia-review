# 多媒体审核任务中心

一个基于FastAPI + Celery的高性能多媒体内容审核系统，支持文档、图片、视频的AI智能审核。

## 🚀 核心特性

- **多模态支持**: 支持PDF、Word、图片、视频等多种文件格式
- **异步处理**: 基于Celery的分布式任务队列，支持高并发处理
- **AI审核**: 集成OCR + 视觉语言模型 + 大语言模型的多层审核
- **实时监控**: 提供任务进度、队列状态的实时监控
- **人工复审**: 支持低置信度结果的人工标注和复审
- **RESTful API**: 完整的REST API，支持前端集成

## 📋 技术架构

```
前端应用
    ↓
FastAPI (API网关)
    ↓
Redis (消息队列 + 缓存)
    ↓
Celery Workers (异步处理)
    ↓
AI服务集群 (OCR + VLLM + LLM)
    ↓
PostgreSQL (数据存储)
```

## 🛠 快速开始

### 1. 环境要求

- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+

### 2. 使用Docker启动（推荐）

```bash
# 克隆项目
git clone <repository-url>
cd multimedia-review-center

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f web
```

### 3. 手动安装

```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export DATABASE_URL="postgresql://user:password@localhost:5432/review_center"
export REDIS_URL="redis://localhost:6379/0"

# 启动Web服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 启动Celery工作器（新终端）
celery -A app.workers.celery_app worker --loglevel=info

# 启动Celery监控（可选）
celery -A app.workers.celery_app flower --port=5555
```

## 📚 API文档

启动服务后访问：
- API文档: http://localhost:8000/docs
- Celery监控: http://localhost:5555
- 健康检查: http://localhost:8000/health

## 🔧 核心API接口

### 任务管理

```bash
# 创建审核任务
POST /api/v1/tasks/
Content-Type: application/json
{
    "name": "测试任务",
    "description": "审核测试文件",
    "strategy_contents": ["涉黄", "暴力", "广告"],
    "video_frame_interval": 5
}

# 获取任务列表
GET /api/v1/tasks/?page=1&size=20&status=pending

# 启动任务处理
POST /api/v1/tasks/{task_id}/start

# 获取任务进度
GET /api/v1/tasks/{task_id}/progress
```

### 文件上传

```bash
# 单文件上传
POST /api/v1/upload/single
Content-Type: multipart/form-data
task_id: {task_id}
file: {file}

# 批量上传
POST /api/v1/upload/batch
Content-Type: multipart/form-data
task_id: {task_id}
files: {file1, file2, ...}

# 查询处理状态
GET /api/v1/upload/status/{file_id}
```

### 审核结果

```bash
# 获取审核结果列表
GET /api/v1/results/?page=1&size=20&violation_type=涉黄

# 获取文件审核结果
GET /api/v1/results/file/{file_id}

# 人工标记结果
POST /api/v1/results/{result_id}/mark
{
    "reviewer_id": "reviewer_001",
    "review_result": "confirmed",
    "review_comment": "确认违规"
}

# 获取待复审列表
GET /api/v1/results/pending-review?priority=confidence
```

## 📊 处理流程

1. **任务创建**: 创建审核任务，配置审核策略
2. **文件上传**: 批量上传待审核文件
3. **任务启动**: 启动异步处理流程
4. **内容提取**: 使用OCR提取文本和图像内容
5. **AI审核**: 
   - 视觉模型审核图像内容
   - 语言模型审核文本内容
6. **结果汇总**: 合并多模型审核结果
7. **人工复审**: 低置信度结果进入人工复审队列
8. **结果输出**: 提供详细的审核报告和违规标注

## 🎯 审核策略

### 支持的违规类型

- **涉黄**: 色情、性暗示内容
- **涉政**: 政治敏感内容
- **暴力**: 血腥、暴力内容  
- **广告**: 商业推广内容
- **违禁词**: 违法违规词汇
- **恐怖主义**: 恐怖主义相关
- **赌博**: 赌博相关内容
- **毒品**: 毒品相关内容

### 智能复审机制

- 置信度 < 0.6 的结果自动进入复审队列
- 政治敏感、恐怖主义内容强制人工复审
- 支持批量标注和复审结果统计

## 📈 监控指标

- 任务处理进度和状态
- 队列长度和等待时间
- 文件处理成功率
- 违规类型分布统计
- 模型置信度分析
- 人工复审效率

## 🔐 安全配置

### 生产环境建议

```bash
# .env 文件
DEBUG=false
DATABASE_URL=postgresql://user:strong_password@db:5432/review_center
REDIS_URL=redis://:redis_password@redis:6379/0

# 文件上传限制
MAX_FILE_SIZE=104857600  # 100MB
ALLOWED_EXTENSIONS=pdf,docx,jpg,png,mp4

# API限流
RATE_LIMIT=100/minute
```

## 🐛 故障排除

### 常见问题

1. **数据库连接失败**
   ```bash
   # 检查数据库状态
   docker-compose ps postgres
   # 查看数据库日志
   docker-compose logs postgres
   ```

2. **Celery任务卡住**
   ```bash
   # 重启工作器
   docker-compose restart worker
   # 检查Redis连接
   docker-compose exec redis redis-cli ping
   ```

3. **文件上传失败**
   ```bash
   # 检查上传目录权限
   ls -la uploads/
   # 检查磁盘空间
   df -h
   ```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 支持

如有问题或建议，请：
- 提交 [Issue](../../issues)
- 发送邮件至：support@example.com
- 查看 [文档中心](./docs/)