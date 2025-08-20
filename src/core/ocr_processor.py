"""
OCR Processor using DOTS.OCR
Based on official implementation from https://github.com/rednote-hilab/dots.ocr
"""

import json
import logging
from typing import Dict, List
from pathlib import Path
import torch
from PIL import Image
import pdf2image

logger = logging.getLogger(__name__)

class DOTSOCRProcessor:
    """
    DOTS.OCR processor for legal documents
    Based on official DotsOCRParser implementation
    """
    
    def __init__(self, model_name: str = None):
        """Initialize DOTS.OCR model"""
        
        # Use the downloaded model path if not specified
        self.model_name = model_name or "/tmp/dots.ocr/weights/DotsOCR"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        # Determine dtype based on device
        # BFloat16 is only efficient on GPUs with proper support
        # On CPU, we must use float32 for compatibility and performance
        if self.device == "cpu":
            self.dtype = torch.float32
            logger.info("Using float32 for CPU inference (BFloat16 not efficient on CPU)")
        else:
            # Even on GPU, we'll use float32 for stability
            self.dtype = torch.float32
            logger.info("Using float32 for stable inference")
        
        # Model parameters
        self.temperature = 0.1
        self.top_p = 1.0
        self.max_new_tokens = 2048  # Reduced for faster processing
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the DOTS.OCR model with proper error handling"""
        try:
            # Monkey patch for torch.compiler compatibility
            if not hasattr(torch, 'compiler'):
                torch.compiler = type('compiler', (), {'is_compiling': lambda: False})()
            elif not hasattr(torch.compiler, 'is_compiling'):
                torch.compiler.is_compiling = lambda: False
            
            # Try to import required modules
            try:
                import sys
                # Add dots.ocr to path for custom modules
                sys.path.insert(0, '/tmp/dots.ocr')
                
                from transformers import AutoModelForCausalLM, AutoProcessor
                from qwen_vl_utils import process_vision_info
                
                # Load model and processor
                logger.info(f"Loading DOTS.OCR model: {self.model_name}")
                
                # Load model (will load with BFloat16 as per config)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,  # Request float32
                    device_map=None,  # Don't use auto device map
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,  # Reduce memory usage
                    attn_implementation="eager"  # Use eager attention to avoid BFloat16 issues
                )
                
                # Aggressively convert ALL model components to float32
                logger.info("Converting model to float32...")
                
                # Convert all parameters
                for param in self.model.parameters():
                    param.data = param.data.to(self.dtype)
                
                # Convert all buffers
                for name, buffer in list(self.model.named_buffers()):
                    if buffer.dtype != self.dtype and torch.is_floating_point(buffer):
                        # Get the parent module and attribute name
                        *parent_path, attr_name = name.split('.')
                        parent = self.model
                        for p in parent_path:
                            parent = getattr(parent, p)
                        # Set the buffer with the new dtype
                        setattr(parent, attr_name, buffer.to(self.dtype))
                
                # Final conversion to ensure everything is in float32
                self.model = self.model.to(dtype=self.dtype, device=self.device)
                
                logger.info(f"Model loaded on {self.device} with dtype {self.dtype}")
                
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    use_fast=True
                )
                
                self.process_vision_info = process_vision_info
                logger.info(f"DOTS.OCR model loaded successfully on {self.device}")
                
            except ImportError as e:
                logger.warning(f"Failed to import required modules: {e}")
                self.model = None
                self.processor = None
                
        except Exception as e:
            logger.error(f"Failed to load DOTS.OCR model: {e}")
            self.model = None
            self.processor = None
    
    async def process(self, document_path: str) -> Dict:
        """
        Process document with DOTS.OCR
        
        Args:
            document_path: Path to document
            
        Returns:
            Dictionary with extracted text, layout, tables, and confidence
        """
        
        # If model is not available, return empty result
        if self.model is None or self.processor is None:
            logger.warning("DOTS.OCR model not available, returning empty result")
            return self._empty_result()
        
        try:
            # Load document images
            images = self._load_document(document_path)
            
            # Process each page
            results = []
            for idx, image in enumerate(images):
                logger.info(f"Processing page {idx + 1}/{len(images)}")
                page_result = self._process_page(image, idx)
                results.append(page_result)
            
            # Combine results
            combined = self._combine_results(results)
            return combined
            
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            return self._empty_result()
    
    def _load_document(self, document_path: str) -> List[Image.Image]:
        """Load document and convert to images"""
        
        path = Path(document_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        if path.suffix.lower() == '.pdf':
            # Convert PDF to images
            images = pdf2image.convert_from_path(document_path, dpi=200)
            logger.info(f"Loaded PDF with {len(images)} pages")
            return images
        else:
            # Load single image
            image = Image.open(document_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return [image]
    
    def _process_page(self, image: Image.Image, page_num: int) -> Dict:
        """Process a single page with DOTS.OCR"""
        
        try:
            # Prepare the prompt for comprehensive extraction
            prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object."""
            
            # Create message format for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process vision information
            image_inputs, video_inputs = self.process_vision_info(messages)
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to device and ensure consistent dtype
            inputs = inputs.to(self.device)
            
            # Convert all floating point tensors to model's dtype
            for key in inputs:
                if torch.is_tensor(inputs[key]) and torch.is_floating_point(inputs[key]):
                    inputs[key] = inputs[key].to(dtype=self.dtype)
            
            # Generate response
            with torch.no_grad():
                # Ensure model is in eval mode
                self.model.eval()
                
                # Generation parameters (do_sample=False for deterministic output)
                generation_kwargs = {
                    "max_new_tokens": self.max_new_tokens,
                    "top_p": self.top_p,
                    "do_sample": False
                }
                
                try:
                    generated_ids = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )
                except RuntimeError as e:
                    if "Input type" in str(e) and "bias type" in str(e):
                        # Log detailed dtype information
                        logger.error(f"Dtype mismatch error: {e}")
                        logger.info("Input dtypes:")
                        for key, value in inputs.items():
                            if torch.is_tensor(value):
                                logger.info(f"  {key}: {value.dtype}")
                        
                        # Try alternative: skip generation and return empty result
                        logger.warning("Skipping OCR due to dtype issues - model may need GPU")
                        return {
                            "text": "[OCR failed - dtype mismatch, GPU recommended]",
                            "page": page_num + 1,
                            "confidence": 0.0,
                            "tables": [],
                            "layout": {"error": "dtype mismatch"}
                        }
                    else:
                        raise
            
            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Parse the response into structured format
            result = {
                "text": response,
                "page": page_num + 1,
                "confidence": 0.9,  # Default confidence
                "tables": [],
                "layout": {"page": page_num + 1}
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}")
            return {
                "text": "",
                "page": page_num + 1,
                "confidence": 0.0,
                "tables": [],
                "layout": {"error": str(e)}
            }
    
    def _combine_results(self, results: List[Dict]) -> Dict:
        """Combine results from multiple pages"""
        
        combined = {
            "text": "",
            "layout": {
                "pages": len(results),
                "structure": []
            },
            "tables": [],
            "reading_order": [],
            "confidence": 0.0,
            "pages": []
        }
        
        # Combine text and other elements
        texts = []
        confidences = []
        
        for result in results:
            if result.get('text'):
                texts.append(result['text'])
            
            if result.get('confidence'):
                confidences.append(result['confidence'])
            
            # Collect tables
            if result.get('tables'):
                combined['tables'].extend(result['tables'])
            
            # Maintain page-level data
            combined['pages'].append({
                "page": result.get('page', 0),
                "confidence": result.get('confidence', 0),
                "table_count": len(result.get('tables', []))
            })
        
        # Join texts with page breaks
        combined['text'] = '\n\n--- Page Break ---\n\n'.join(texts) if texts else ""
        
        # Calculate average confidence
        if confidences:
            combined['confidence'] = sum(confidences) / len(confidences)
        
        # Build reading order
        combined['reading_order'] = texts
        
        return combined
    
    def _empty_result(self) -> Dict:
        """Return empty result when processing fails"""
        return {
            "text": "",
            "layout": {"pages": 0, "structure": []},
            "tables": [],
            "reading_order": [],
            "confidence": 0.0,
            "pages": []
        }
    
    def cleanup(self):
        """Clean up resources"""
        
        try:
            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Delete model to free memory
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor
            
            logger.info("OCR processor cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")