"""
OCR识别服务
负责调用Dots OCR API进行文本和图像识别
"""

import httpx
import json
import base64
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import asyncio
from loguru import logger
from app.config import get_settings
from app.utils.file_utils import FileUtils


class OCRService:
    """OCR识别服务类"""
    
    def __init__(self):
        self.settings = get_settings()
        self.api_url = self.settings.OCR_API_URL
        self.timeout = 30.0  # 请求超时时间
    
    async def extract_content(self, image_path: str) -> Dict:
        """
        使用AI模型从图片中提取文本和图像块
        """
        try:
            # 将图片编码为base64
            image_base64 = self._encode_image_to_base64(image_path)
            if not image_base64:
                return {"success": False, "error": "图片编码失败"}
            
            # 构建OCR提示词
            prompt = """
    请分析这张图片，提取其中的文字内容和图像区域。

    返回JSON格式：
    {
        "text_blocks": [
            {
                "text": "识别的文字内容",
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.95
            }
        ],
        "image_blocks": [
            {
                "description": "图像内容描述",
                "bbox": [x1, y1, x2, y2]
            }
        ]
    }

    要求：
    1. 准确识别所有可见文字
    2. 标注文字和图像的位置坐标
    3. 如果没有文字则text_blocks为空数组
    """
            
            # 准备请求数据
            payload = {
                "model": self.settings.OPENROUTER_VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                "max_tokens": 1500,
                "temperature": 0.1
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
                    return self._process_ai_ocr_result(result, image_path)
                else:
                    return {"success": False, "error": f"AI OCR调用失败: {response.status_code}"}
        
        except Exception as e:
            return {"success": False, "error": f"AI OCR识别异常: {str(e)}"}

    def _process_ai_ocr_result(self, api_result: Dict, image_path: str) -> Dict:
        """处理AI OCR结果"""
        try:
            if "choices" in api_result and len(api_result["choices"]) > 0:
                content = api_result["choices"][0]["message"]["content"]
                
                # 解析JSON内容
                try:
                    result_data = json.loads(content)
                except json.JSONDecodeError:
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        result_data = json.loads(json_match.group())
                    else:
                        return {"success": False, "error": "无法解析AI OCR结果"}
                
                # 转换为标准格式
                blocks = []
                
                # 处理文字块
                for text_block in result_data.get("text_blocks", []):
                    blocks.append({
                        "type": "text",
                        "text": text_block.get("text", ""),
                        "bbox": text_block.get("bbox", [0, 0, 0, 0]),
                        "confidence": text_block.get("confidence", 0.9)
                    })
                
                # 处理图像块
                for img_block in result_data.get("image_blocks", []):
                    blocks.append({
                        "type": "image",
                        "image_path": image_path,  # 使用原图路径
                        "bbox": img_block.get("bbox", [0, 0, 0, 0]),
                        "description": img_block.get("description", "")
                    })
                
                return {
                    "success": True,
                    "blocks": blocks,
                    "total_text_blocks": len([b for b in blocks if b["type"] == "text"]),
                    "total_image_blocks": len([b for b in blocks if b["type"] == "image"])
                }
        
        except Exception as e:
            return {"success": False, "error": f"AI OCR结果处理失败: {str(e)}"}
        
    #not use
    async def extract_content_ocr_api(self, image_path: str) -> Dict:
        """
        从图片中提取文本和图像块
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            OCR识别结果，格式：
            {
                "success": True,
                "blocks": [
                    {
                        "type": "text",
                        "text": "识别的文字内容", 
                        "bbox": [x1, y1, x2, y2],
                        "confidence": 0.95
                    },
                    {
                        "type": "image",
                        "image_path": "/tmp/extracted_image.jpg",
                        "bbox": [x1, y1, x2, y2],
                        "description": "图像内容描述"
                    }
                ]
            }
        """
        try:
            # 将图片转换为base64编码
            image_base64 = self._encode_image_to_base64(image_path)
            if not image_base64:
                return {"success": False, "error": "图片编码失败"}
            
            # 准备请求数据
            payload = {
                "image": image_base64,
                "options": {
                    "extract_text": True,      # 提取文字
                    "extract_images": True,    # 提取图像块
                    "return_confidence": True, # 返回置信度
                    "language": "auto"         # 自动检测语言
                }
            }
            
            # 调用OCR API
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_url}/extract",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return self._process_ocr_result(result, image_path)
                else:
                    return {
                        "success": False, 
                        "error": f"OCR API调用失败: {response.status_code}"
                    }
        
        except httpx.TimeoutException:
            return {"success": False, "error": "OCR API请求超时"}
        except Exception as e:
            return {"success": False, "error": f"OCR识别异常: {str(e)}"}
    
    async def extract_from_document(self, doc_path: str) -> List[Dict]:
        """
        从文档中提取内容（需要先转换为图片）
        
        Args:
            doc_path: 文档文件路径
            
        Returns:
            每页的OCR识别结果列表
        """
        try:
            # 将文档转换为图片列表
            image_paths = await self._convert_document_to_images(doc_path)
            if not image_paths:
                return []
            
            # 并发处理所有页面
            tasks = []
            for page_num, image_path in enumerate(image_paths, 1):
                task = self._extract_page_content(image_path, page_num)
                tasks.append(task)
            
            # 等待所有页面处理完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 过滤异常结果
            valid_results = []
            for result in results:
                if isinstance(result, dict) and result.get("success"):
                    valid_results.append(result)
            
            # 清理临时图片文件
            FileUtils.cleanup_temp_files(image_paths)
            
            return valid_results
        
        except Exception as e:
            logger.info(f"文档OCR处理失败: {e}")
            return []
    
    async def extract_from_video_frames(self, frame_paths: List[str]) -> List[Dict]:
        """
        从视频帧中提取内容
        
        Args:
            frame_paths: 视频帧图片路径列表
            
        Returns:
            每帧的OCR识别结果列表
        """
        try:
            # 并发处理所有帧
            tasks = []
            for frame_num, frame_path in enumerate(frame_paths):
                task = self._extract_frame_content(frame_path, frame_num)
                tasks.append(task)
            
            # 等待所有帧处理完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 过滤有效结果
            valid_results = []
            for result in results:
                if isinstance(result, dict) and result.get("success"):
                    valid_results.append(result)
            
            return valid_results
        
        except Exception as e:
            logger.info(f"视频帧OCR处理失败: {e}")
            return []
    
    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """将图片编码为base64字符串"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                return encoded_string.decode('utf-8')
        except Exception as e:
            logger.info(f"图片编码失败: {e}")
            return None
    
    def _process_ocr_result(self, api_result: Dict, image_path: str) -> Dict:
        """
        处理OCR API返回结果，转换为标准格式
        
        Args:
            api_result: OCR API原始返回结果
            image_path: 源图片路径
            
        Returns:
            标准化的OCR结果
        """
        try:
            blocks = []
            
            # 处理文字块
            if "text_blocks" in api_result:
                for text_block in api_result["text_blocks"]:
                    blocks.append({
                        "type": "text",
                        "text": text_block.get("text", ""),
                        "bbox": text_block.get("bbox", [0, 0, 0, 0]),
                        "confidence": text_block.get("confidence", 0.0)
                    })
            
            # 处理图像块
            if "image_blocks" in api_result:
                for img_block in api_result["image_blocks"]:
                    # 保存提取的图像块
                    image_block_path = self._save_image_block(
                        img_block.get("image_data"),
                        image_path
                    )
                    
                    blocks.append({
                        "type": "image", 
                        "image_path": image_block_path,
                        "bbox": img_block.get("bbox", [0, 0, 0, 0]),
                        "description": img_block.get("description", "")
                    })
            
            return {
                "success": True,
                "blocks": blocks,
                "total_text_blocks": len([b for b in blocks if b["type"] == "text"]),
                "total_image_blocks": len([b for b in blocks if b["type"] == "image"])
            }
        
        except Exception as e:
            return {"success": False, "error": f"结果处理失败: {str(e)}"}
    
    def _save_image_block(self, image_data: str, source_path: str) -> str:
        """保存提取的图像块"""
        try:
            import uuid
            import os
            
            # 解码base64图像数据
            image_bytes = base64.b64decode(image_data)
            
            # 生成保存路径
            temp_dir = os.path.join(self.settings.UPLOAD_DIR, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            image_block_path = os.path.join(
                temp_dir, 
                f"img_block_{uuid.uuid4()}.jpg"
            )
            
            # 保存图像
            with open(image_block_path, "wb") as f:
                f.write(image_bytes)
            
            return image_block_path
        
        except Exception as e:
            logger.info(f"保存图像块失败: {e}")
            return ""
    
    async def _convert_document_to_images(self, doc_path: str) -> List[str]:
        """将文档转换为图片列表"""
        try:
            from pdf2image import convert_from_path
            import uuid
            import os
            
            # 创建临时目录
            temp_dir = os.path.join(
                self.settings.UPLOAD_DIR, 
                "temp", 
                f"doc_{uuid.uuid4()}"
            )
            os.makedirs(temp_dir, exist_ok=True)
            
            file_ext = Path(doc_path).suffix.lower()
            image_paths = []
            
            if file_ext == '.pdf':
                # PDF转图片
                images = convert_from_path(doc_path, dpi=200)
                for i, image in enumerate(images):
                    image_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
                    image.save(image_path, 'JPEG')
                    image_paths.append(image_path)
            
            elif file_ext in ['.docx', '.doc']:
                # Word文档转换（需要额外的库支持）
                # 这里简化处理，实际项目中可能需要调用LibreOffice等工具
                logger.info("Word文档转换暂未实现")
                
            return image_paths
        
        except Exception as e:
            logger.info(f"文档转图片失败: {e}")
            return []
    
    async def _extract_page_content(self, image_path: str, page_num: int) -> Dict:
        """提取单页内容"""
        result = await self.extract_content(image_path)
        if result.get("success"):
            result["page_number"] = page_num
        return result
    
    async def _extract_frame_content(self, frame_path: str, frame_num: int) -> Dict:
        """提取单帧内容"""
        result = await self.extract_content(frame_path)
        if result.get("success"):
            result["frame_number"] = frame_num
        return result
    
    async def batch_extract(self, image_paths: List[str]) -> List[Dict]:
        """
        批量OCR识别
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            批量识别结果
        """
        try:
            # 限制并发数量避免过载
            semaphore = asyncio.Semaphore(5)  # 最多5个并发请求
            
            async def extract_with_semaphore(path):
                async with semaphore:
                    return await self.extract_content(path)
            
            # 并发处理
            tasks = [extract_with_semaphore(path) for path in image_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 过滤有效结果
            valid_results = []
            for result in results:
                if isinstance(result, dict) and result.get("success"):
                    valid_results.append(result)
            
            return valid_results
        
        except Exception as e:
            logger.info(f"批量OCR处理失败: {e}")
            return []