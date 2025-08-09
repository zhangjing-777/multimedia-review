-- 1. 创建审核任务表
CREATE TABLE review_tasks (
    id UUID PRIMARY KEY ,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    strategy_type VARCHAR(100),
    strategy_contents TEXT,
    video_frame_interval INTEGER DEFAULT 5,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    error_message TEXT,
    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    violation_count INTEGER DEFAULT 0,
    creator_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- 添加注释
COMMENT ON TABLE review_tasks IS '审核任务表';
COMMENT ON COLUMN review_tasks.id IS '任务唯一标识符';
COMMENT ON COLUMN review_tasks.name IS '任务名称';
COMMENT ON COLUMN review_tasks.description IS '任务描述';
COMMENT ON COLUMN review_tasks.strategy_type IS '审核策略类型';
COMMENT ON COLUMN review_tasks.strategy_contents IS '审核策略内容';
COMMENT ON COLUMN review_tasks.video_frame_interval IS '视频抽帧间隔(秒)';
COMMENT ON COLUMN review_tasks.status IS '任务状态';
COMMENT ON COLUMN review_tasks.progress IS '处理进度(0-100)';
COMMENT ON COLUMN review_tasks.error_message IS '错误信息';
COMMENT ON COLUMN review_tasks.total_files IS '总文件数';
COMMENT ON COLUMN review_tasks.processed_files IS '已处理文件数';
COMMENT ON COLUMN review_tasks.violation_count IS '违规文件数量';
COMMENT ON COLUMN review_tasks.creator_id IS '创建者ID';
COMMENT ON COLUMN review_tasks.created_at IS '创建时间';
COMMENT ON COLUMN review_tasks.updated_at IS '更新时间';
COMMENT ON COLUMN review_tasks.started_at IS '开始处理时间';
COMMENT ON COLUMN review_tasks.completed_at IS '完成时间';

-- 2. 创建审核文件表
CREATE TABLE review_files (
    id UUID PRIMARY KEY ,
    task_id UUID NOT NULL,
    original_name VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000) NOT NULL,
    file_type VARCHAR(20) NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type VARCHAR(100),
    file_extension VARCHAR(10),
    content_hash VARCHAR(64),
    page_count INTEGER,
    duration INTEGER,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    error_message TEXT,
    ocr_blocks_count INTEGER DEFAULT 0,
    text_blocks_count INTEGER DEFAULT 0,
    image_blocks_count INTEGER DEFAULT 0,
    violation_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    
    -- 外键约束
    CONSTRAINT fk_review_files_task_id 
        FOREIGN KEY (task_id) 
        REFERENCES review_tasks(id) 
        ON DELETE CASCADE
);

-- 添加注释
COMMENT ON TABLE review_files IS '审核文件表';
COMMENT ON COLUMN review_files.id IS '文件唯一标识符';
COMMENT ON COLUMN review_files.task_id IS '所属任务ID';
COMMENT ON COLUMN review_files.original_name IS '原始文件名';
COMMENT ON COLUMN review_files.file_path IS '文件存储路径';
COMMENT ON COLUMN review_files.file_type IS '文件类型';
COMMENT ON COLUMN review_files.file_size IS '文件大小(字节)';
COMMENT ON COLUMN review_files.mime_type IS 'MIME类型';
COMMENT ON COLUMN review_files.file_extension IS '文件扩展名';
COMMENT ON COLUMN review_files.content_hash IS '文件内容MD5哈希值，用于去重';
COMMENT ON COLUMN review_files.page_count IS '页数/帧数（文档/视频）';
COMMENT ON COLUMN review_files.duration IS '时长(秒)，仅视频文件';
COMMENT ON COLUMN review_files.status IS '处理状态';
COMMENT ON COLUMN review_files.progress IS '处理进度(0-100)';
COMMENT ON COLUMN review_files.error_message IS '错误信息';
COMMENT ON COLUMN review_files.ocr_blocks_count IS 'OCR识别的内容块数量';
COMMENT ON COLUMN review_files.text_blocks_count IS '文本块数量';
COMMENT ON COLUMN review_files.image_blocks_count IS '图像块数量';
COMMENT ON COLUMN review_files.violation_count IS '违规内容数量';
COMMENT ON COLUMN review_files.created_at IS '创建时间';
COMMENT ON COLUMN review_files.updated_at IS '更新时间';
COMMENT ON COLUMN review_files.processed_at IS '处理完成时间';

-- 3. 创建审核结果表
CREATE TABLE review_results (
    id UUID PRIMARY KEY ,
    file_id UUID NOT NULL,
    violation_type VARCHAR(20) NOT NULL,
    source_type VARCHAR(20) NOT NULL,
    confidence_score REAL NOT NULL DEFAULT 0.0,
    evidence TEXT,
    evidence_text TEXT,
    position JSONB,
    page_number INTEGER,
    timestamp REAL,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    raw_response JSONB,
    is_reviewed BOOLEAN DEFAULT FALSE,
    reviewer_id VARCHAR(100),
    review_result VARCHAR(20),
    review_comment TEXT,
    review_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 外键约束
    CONSTRAINT fk_review_results_file_id 
        FOREIGN KEY (file_id) 
        REFERENCES review_files(id) 
        ON DELETE CASCADE
);

-- 添加注释
COMMENT ON TABLE review_results IS '审核结果表';
COMMENT ON COLUMN review_results.id IS '结果唯一标识符';
COMMENT ON COLUMN review_results.file_id IS '所属文件ID';
COMMENT ON COLUMN review_results.violation_type IS '违规类型';
COMMENT ON COLUMN review_results.source_type IS '识别来源';
COMMENT ON COLUMN review_results.confidence_score IS '置信度分数(0-1)';
COMMENT ON COLUMN review_results.evidence IS '违规证据/内容描述';
COMMENT ON COLUMN review_results.evidence_text IS '违规文本内容';
COMMENT ON COLUMN review_results.position IS '位置信息，格式：{"page": 1, "bbox": [x1,y1,x2,y2]} 或 {"timestamp": 120}';
COMMENT ON COLUMN review_results.page_number IS '页码（文档）或帧序号（视频）';
COMMENT ON COLUMN review_results.timestamp IS '时间戳（视频，单位：秒）';
COMMENT ON COLUMN review_results.model_name IS '使用的AI模型名称';
COMMENT ON COLUMN review_results.model_version IS '模型版本';
COMMENT ON COLUMN review_results.raw_response IS 'AI模型原始返回结果';
COMMENT ON COLUMN review_results.is_reviewed IS '是否已人工复审';
COMMENT ON COLUMN review_results.reviewer_id IS '复审人员ID';
COMMENT ON COLUMN review_results.review_result IS '人工复审结果：confirmed/rejected/modified';
COMMENT ON COLUMN review_results.review_comment IS '复审备注';
COMMENT ON COLUMN review_results.review_time IS '复审时间';
COMMENT ON COLUMN review_results.created_at IS '创建时间';
COMMENT ON COLUMN review_results.updated_at IS '更新时间';

-- 创建索引优化查询性能
CREATE INDEX idx_review_files_task_id ON review_files(task_id);
CREATE INDEX idx_review_files_status ON review_files(status);
CREATE INDEX idx_review_files_created_at ON review_files(created_at);

CREATE INDEX idx_review_results_file_id ON review_results(file_id);
CREATE INDEX idx_review_results_violation_type ON review_results(violation_type);
CREATE INDEX idx_review_results_is_reviewed ON review_results(is_reviewed);
CREATE INDEX idx_review_results_created_at ON review_results(created_at);

CREATE INDEX idx_review_tasks_status ON review_tasks(status);
CREATE INDEX idx_review_tasks_created_at ON review_tasks(created_at);