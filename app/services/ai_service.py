"""
AI审核服务
负责调用VLLM和LLM进行内容审核
"""

import httpx
import json
import base64
from typing import List, Dict, Optional
import asyncio
from loguru import logger
from app.config import get_settings
from app.models.result import ViolationType, SourceType


class AIReviewService:
    """AI审核服务类"""
    
    def __init__(self):
        self.settings = get_settings()
        self.timeout = 60.0  # AI服务超时时间较长
    
   
    async def review_visual_content(
        self, 
        image_path: str, 
        strategy_type: str = None,
        strategy_contents: str = None
    ) -> List[Dict]:
        """
        使用OpenRouter视觉语言模型审核图像内容
        
        Args:
            image_path: 图片路径
            strategy_type: 审核策略类型
            strategy_contents: 审核策略内容
            
        Returns:
            视觉审核结果列表
        """
        try:
            # 将图片编码为base64
            image_base64 = self._encode_image_to_base64(image_path)
            if not image_base64:
                return []
            
            # 构建审核提示词
            prompt = self._build_visual_review_prompt(strategy_type, strategy_contents)
            
            # OpenRouter API payload - 使用OpenAI格式
            payload = {
                "model": self.settings.OPENROUTER_VISION_MODEL,  
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.1,
                # OpenRouter特有参数
                "top_p": 0.9
            }
            
            headers = {
                "Authorization": f"Bearer {self.settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # 调用OpenRouter API
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.settings.ENDPOINT,
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return self._process_visual_result(result, image_path)
                else:
                    error_detail = response.text
                    logger.info(f"OpenRouter视觉API调用失败: {response.status_code} - {error_detail}")
                    return []
    
        except httpx.TimeoutException:
            logger.info("OpenRouter视觉API请求超时")
            return []
        except Exception as e:
            logger.info(f"视觉内容审核失败: {e}")
            return []


    async def review_text_content(
        self, 
        text_content: str, 
        strategy_type: str = None,
        strategy_contents: str = None
    ) -> List[Dict]:
        """
        使用OpenRouter大语言模型审核文本内容
        
        Args:
            text_content: 文本内容
            strategy_type: 审核策略类型
            strategy_contents: 审核策略内容
            
        Returns:
            文本审核结果列表
        """
        try:
            # 构建文本审核提示词
            prompt = self._build_text_review_prompt(text_content, strategy_type, strategy_contents)
            
            # OpenRouter API payload
            payload = {
                "model": self.settings.OPENROUTER_TEXT_MODEL,  
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的内容审核AI，请严格按照给定的JSON格式返回审核结果。"
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.1,
                "top_p": 0.9,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
            
            # 请求头
            headers = {
                "Authorization": f"Bearer {self.settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # 调用OpenRouter API
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.settings.ENDPOINT,
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return self._process_text_result(result, text_content)
                else:
                    error_detail = response.text
                    logger.info(f"OpenRouter文本API调用失败: {response.status_code} - {error_detail}")
                    return []
        
        except httpx.TimeoutException:
            logger.info("OpenRouter文本API请求超时")
            return []
        except Exception as e:
            logger.info(f"文本内容审核失败: {e}")
            return []

   
    async def batch_review_images(
        self, 
        image_paths: List[str], 
        strategy_contents: List[str]
    ) -> List[Dict]:
        """
        批量审核图像内容
        
        Args:
            image_paths: 图片路径列表
            strategy_contents: 审核策略
            
        Returns:
            批量审核结果
        """
        try:
            # 限制并发数量
            semaphore = asyncio.Semaphore(3)  # VLLM服务负载较高，限制并发
            
            async def review_with_semaphore(path):
                async with semaphore:
                    return await self.review_visual_content(path, strategy_contents)
            
            # 并发处理
            tasks = [review_with_semaphore(path) for path in image_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 合并结果
            all_violations = []
            for result in results:
                if isinstance(result, list):
                    all_violations.extend(result)
            
            return all_violations
        
        except Exception as e:
            logger.info(f"批量图像审核失败: {e}")
            return []
    
    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """将图片编码为base64"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                return encoded_string.decode('utf-8')
        except Exception as e:
            logger.info(f"图片编码失败: {e}")
            return None
        
    def _build_visual_review_prompt(self, strategy_type: str = None, strategy_contents: str = None) -> str:
        """构建视觉审核提示词"""
        
        # 如果参数为空，使用默认的全面检测内容
        if not strategy_type and not strategy_contents:
            detection_content = """【一、文本违规类型】
    - 涉黄：色情描述、低俗交友、引流链接
    - 涉政：敏感人物、政治事件、极端言论
    - 涉恐：暴力威胁、爆炸袭击、恐吓语言
    - 恶意辱骂：人身攻击、网暴言论、侮辱性词语
    - 虚假诈骗：诱导、假冒、欺诈套路
    - 灌水低质：乱码、刷屏、无意义内容
    - 敏感媒体：境外媒体/账号、敏感来源
    - 广告推广：促销用语、二维码引流、诱导转化
    - 违禁信息：毒品、枪支、暴力组织
    - 舆情极端：煽动仇恨、煽动群体对立
    - 黑名单词命中：如命中企业定制敏感词库
    - 地图错误：涉"台独"、"港独"、国界错误
    - 劣迹艺人相关内容

    【二、图像违规类型】
    - 涉黄/涉政/暴恐/恶心类图片
    - 广告图、二维码、违禁旗帜
    - 人物脸、logo、水印等识别
    - OCR识别后含敏感文本
    - 模糊遮挡、伪造或修改图
    - 命中定制黑库/白库
    - 错误地图/敏感图形
    - EasyDL 风险特征图（如异常弹窗、黑警帽等）
    - 暴力/危险现场图（如丧尸、袭击、打斗、事故现场、武器等）

    【三、人物违规类型】
    - 如果图片中出现人物，请描述其外观特征（性别、年龄段、着装等）
    - 注意是否为公众人物、艺人或网络名人
    - 检查是否存在争议人物、政治敏感人物、劣迹艺人的可能性
    - 如果无法确定人物身份但怀疑可能有问题，请标记"不确定"

    【四、法律违规类型】
    - 违反《广告法》或《反不正当竞争法》"""
        else:
            # 使用自定义的审核策略
            detection_content = f"内容类型：{strategy_type or '未指定'}\n审核策略：{strategy_contents or '未指定'}"
        
        prompt = f"""
    你是一个专业的内容审核AI，请仔细分析这张图片，根据以下审核标准检测是否包含违规内容：

    {detection_content}

    请按照以下JSON格式返回分析结果：
    {{
        "violations": [
            {{
                "type": "违规类型",
                "confidence": 0.95,
                "evidence": "具体的违规证据描述",
                "bbox": [x1, y1, x2, y2]  // 可选，违规区域坐标
            }}
        ],
        "is_safe": true/false,
        "analysis": "整体分析说明"
    }}

    如果图片安全无违规，violations数组为空，is_safe为true。
    置信度范围0-1，只有置信度>0.5的结果才报告。
    """
        return prompt

    def _build_text_review_prompt(self, text: str, strategy_type: str = None, strategy_contents: str = None) -> str:
        """构建文本审核提示词"""
        
        # 如果参数为空，使用默认的全面检测内容
        if not strategy_type and not strategy_contents:
            detection_content = """【一、文本违规类型】
    - 涉黄：色情描述、低俗交友、引流链接
    - 涉政：敏感人物、政治事件、极端言论
    - 涉恐：暴力威胁、爆炸袭击、恐吓语言
    - 恶意辱骂：人身攻击、网暴言论、侮辱性词语
    - 虚假诈骗：诱导、假冒、欺诈套路
    - 灌水低质：乱码、刷屏、无意义内容
    - 敏感媒体：境外媒体/账号、敏感来源
    - 广告推广：促销用语、二维码引流、诱导转化
    - 违禁信息：毒品、枪支、暴力组织
    - 舆情极端：煽动仇恨、煽动群体对立
    - 黑名单词命中：如命中企业定制敏感词库
    - 地图错误：涉"台独"、"港独"、国界错误
    - 劣迹艺人相关内容

    【二、图像违规类型】
    - 涉黄/涉政/暴恐/恶心类图片
    - 广告图、二维码、违禁旗帜
    - 人物脸、logo、水印等识别
    - OCR识别后含敏感文本
    - 模糊遮挡、伪造或修改图
    - 命中定制黑库/白库
    - 错误地图/敏感图形
    - EasyDL 风险特征图（如异常弹窗、黑警帽等）
    - 暴力/危险现场图（如丧尸、袭击、打斗、事故现场、武器等）

    【三、人物违规类型】
    - 如果图片中出现人物，请描述其外观特征（性别、年龄段、着装等）
    - 注意是否为公众人物、艺人或网络名人
    - 检查是否存在争议人物、政治敏感人物、劣迹艺人的可能性
    - 如果无法确定人物身份但怀疑可能有问题，请标记"不确定"

    【四、法律违规类型】
    - 违反《广告法》或《反不正当竞争法》"""
        else:
            # 使用自定义的审核策略
            detection_content = f"内容类型：{strategy_type or '未指定'}\n审核策略：{strategy_contents or '未指定'}"
        
        prompt = f"""
    你是一个专业的文本内容审核AI，请分析以下文本内容，根据审核标准检测是否包含违规信息：

    待审核文本：
    \"\"\"
    {text[:2000]}  # 限制文本长度避免超出模型限制
    \"\"\"

    审核标准：
    {detection_content}

    请按照以下JSON格式返回分析结果：
    {{
        "violations": [
            {{
                "type": "违规类型",
                "confidence": 0.95,
                "evidence": "具体的违规文本片段",
                "start_pos": 100,  // 可选，违规文本起始位置
                "end_pos": 150     // 可选，违规文本结束位置
            }}
        ],
        "is_safe": true/false,
        "analysis": "整体分析说明",
        "keywords": ["违规关键词1", "违规关键词2"]  // 可选
    }}

    要求：
    1. 只报告置信度>0.6的违规内容
    2. evidence要包含具体的违规文本片段
    3. 分析要客观准确，避免误报
    """
        return prompt
               

    def _process_visual_result(self, api_result: Dict, image_path: str) -> List[Dict]:
        """处理OpenRouter视觉审核结果"""
        violations = []
        
        try:
            # OpenRouter返回格式：choices[0].message.content
            if "choices" in api_result and len(api_result["choices"]) > 0:
                content = api_result["choices"][0]["message"]["content"]
                
                # 尝试解析JSON内容
                try:
                    # 如果返回的是纯JSON字符串
                    result_data = json.loads(content)
                except json.JSONDecodeError:
                    # 如果返回的内容包含其他文本，尝试提取JSON部分
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        result_data = json.loads(json_match.group())
                    else:
                        logger.info(f"无法解析视觉审核结果: {content}")
                        return []
                
                # 处理违规结果
                for violation in result_data.get("violations", []):
                    violations.append({
                        "violation_type": violation.get("type"),
                        "source_type": SourceType.VISUAL,
                        "confidence_score": float(violation.get("confidence", 0.0)),
                        "evidence": violation.get("evidence", ""),
                        "position": {
                            "bbox": violation.get("bbox", []),
                            "image_path": image_path
                        },
                        "model_name": api_result.get("model", "unknown"),
                        "raw_response": api_result
                    })
            
        except Exception as e:
            logger.info(f"视觉结果处理失败: {e}")
        
        return violations


    def _process_text_result(self, api_result: Dict, text_content: str) -> List[Dict]:
        """处理OpenRouter文本审核结果"""
        violations = []
        
        try:
            # OpenRouter返回格式：choices[0].message.content
            if "choices" in api_result and len(api_result["choices"]) > 0:
                content = api_result["choices"][0]["message"]["content"]
                
                # 尝试解析JSON内容
                try:
                    result_data = json.loads(content)
                except json.JSONDecodeError:
                    # 尝试提取JSON部分
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        result_data = json.loads(json_match.group())
                    else:
                        logger.info(f"无法解析文本审核结果: {content}")
                        return []
                
                # 处理违规结果
                for violation in result_data.get("violations", []):
                    violations.append({
                        "violation_type": violation.get("type"),
                        "source_type": SourceType.OCR,
                        "confidence_score": float(violation.get("confidence", 0.0)),
                        "evidence": violation.get("evidence", ""),
                        "evidence_text": violation.get("evidence", ""),
                        "position": {
                            "start_pos": violation.get("start_pos"),
                            "end_pos": violation.get("end_pos")
                        },
                        "model_name": api_result.get("model", "unknown"),
                        "raw_response": api_result
                    })
            
        except Exception as e:
            logger.info(f"文本结果处理失败: {e}")
        
        return violations