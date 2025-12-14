"""
Generate GPT-4o predictions for ViVQA Test Set

Run GPT-4o API calls and save results to JSONL file.
Supports caching and resume for quota management.

Usage:
    export OPENAI_API_KEY="sk-your-key"
    python generate_gpt4o_predictions.py
"""

import os
import json
import time
import pandas as pd
from tqdm import tqdm
from PIL import Image
import base64
from io import BytesIO

# OpenAI API
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("[ERROR] openai package not installed. Run: pip install openai")
    OPENAI_AVAILABLE = False

# ======================
# CONFIG
# ======================
TEST_CSV_PATH = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/test.csv"
IMAGE_FOLDER = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/test"
OUTPUT_JSONL = "/kaggle/working/gpt4o_predictions.jsonl"  # Output predictions

# API Settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL = "gpt-4o"  # or "gpt-4o-mini"
MAX_TOKENS = 150
TEMPERATURE = 0.1
TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2

# Rate limiting (Free tier with data sharing)
MAX_TOKENS_PER_DAY = 250000  # GPT-4o: 250K tokens/day
MAX_TOKENS_PER_MINUTE = 10000
MAX_REQUESTS_PER_MINUTE = 200
RATE_LIMIT_DELAY = 0.5  # seconds between requests
MAX_SAMPLES = None  # None = process all

# ======================
# RATE LIMITER
# ======================
class RateLimiter:
    """Track token usage and enforce rate limits"""
    def __init__(self, max_tokens_per_day, max_tokens_per_minute):
        self.max_tokens_per_day = max_tokens_per_day
        self.max_tokens_per_minute = max_tokens_per_minute
        
        self.request_count = 0
        self.tokens_this_minute = 0
        self.tokens_today = 0
        self.minute_start_time = time.time()
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def check_and_wait(self, estimated_tokens=1500):
        """Check limits and wait if necessary"""
        current_time = time.time()
        
        # Reset minute counter
        if current_time - self.minute_start_time >= 60:
            self.tokens_this_minute = 0
            self.minute_start_time = current_time
        
        # Check daily token limit
        if self.tokens_today + estimated_tokens > self.max_tokens_per_day:
            print(f"\n[WARN] Daily token limit reached ({self.max_tokens_per_day:,} tokens)")
            print(f"[INFO] Used {self.tokens_today:,} tokens today")
            print(f"[INFO] Stopping to avoid exceeding quota")
            return False
        
        # Check tokens per minute limit
        if self.tokens_this_minute + estimated_tokens > self.max_tokens_per_minute:
            wait_time = 60 - (current_time - self.minute_start_time)
            if wait_time > 0:
                print(f"\n[WARN] Token/min limit approaching, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.tokens_this_minute = 0
                self.minute_start_time = time.time()
        
        return True
    
    def record_request(self, input_tokens=0, output_tokens=0):
        """Record a completed request"""
        self.request_count += 1
        total_tokens = input_tokens + output_tokens
        self.tokens_this_minute += total_tokens
        self.tokens_today += total_tokens
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
    
    def get_stats(self):
        """Get usage statistics"""
        return {
            'requests': self.request_count,
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens
        }

# ======================
# IMAGE ENCODING
# ======================
def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 for GPT-4o"""
    try:
        with Image.open(image_path) as img:
            # Resize if too large
            max_size = 1024
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Encode to base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"[ERROR] Failed to encode {image_path}: {e}")
        return None

# ======================
# GPT-4o QUERY
# ======================
def query_gpt4o(client, image_base64: str, question: str, rate_limiter: RateLimiter) -> dict:
    """Query GPT-4o and return raw response"""
    if not image_base64:
        return {
            'raw_output': '',
            'error': 'encoding_error',
            'input_tokens': 0,
            'output_tokens': 0
        }
    
    # Prompt
    prompt = f"""Bạn là một trợ lý AI chuyên trả lời câu hỏi về hình ảnh bằng tiếng Việt.

Câu hỏi: {question}

Hãy trả lời theo format sau:
Answer: [Câu trả lời ngắn gọn]
Reasoning: [Giải thích lý do tại sao bạn trả lời như vậy, dựa trên các chi tiết trong hình]

Lưu ý:
- Trả lời bằng tiếng Việt
- Câu trả lời phải chính xác và cụ thể
- Giải thích phải dựa vào các chi tiết quan sát được trong hình"""

    # Check rate limits
    if not rate_limiter.check_and_wait(1500):
        return {
            'raw_output': '',
            'error': 'rate_limit_reached',
            'input_tokens': 0,
            'output_tokens': 0
        }

    # API call with retry
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_base64,
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                timeout=TIMEOUT
            )
            
            # Extract response
            raw_text = response.choices[0].message.content.strip()
            
            # Get token usage
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
            
            # Record usage
            rate_limiter.record_request(input_tokens, output_tokens)
            
            return {
                'raw_output': raw_text,
                'error': '',
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if rate limit error
            if 'rate_limit' in error_msg.lower() or 'quota' in error_msg.lower():
                print(f"\n[ERROR] API rate limit: {error_msg}")
                return {
                    'raw_output': '',
                    'error': 'api_rate_limit',
                    'input_tokens': 0,
                    'output_tokens': 0
                }
            
            print(f"[WARN] Attempt {attempt+1}/{MAX_RETRIES} failed: {error_msg}")
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return {
                    'raw_output': '',
                    'error': error_msg,
                    'input_tokens': 0,
                    'output_tokens': 0
                }

# ======================
# CACHE MANAGEMENT
# ======================
def load_predictions(jsonl_path: str) -> dict:
    """Load existing predictions"""
    cache = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                key = (str(data['img_id']), str(data['question']))
                cache[key] = data
        print(f"[INFO] Loaded {len(cache)} cached predictions")
    return cache

def save_prediction(jsonl_path: str, record: dict):
    """Append prediction to JSONL"""
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

# ======================
# MAIN
# ======================
def main():
    print("="*70)
    print("GPT-4o PREDICTION GENERATION FOR ViVQA TEST SET")
    print("="*70)
    
    # Check API key
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not set!")
        print("Set it via: export OPENAI_API_KEY='sk-your-key'")
        return
    
    if not OPENAI_AVAILABLE:
        return
    
    # Initialize
    client = OpenAI(api_key=OPENAI_API_KEY)
    rate_limiter = RateLimiter(MAX_TOKENS_PER_DAY, MAX_TOKENS_PER_MINUTE)
    
    print(f"[INFO] Model: {GPT_MODEL}")
    print(f"[INFO] Rate limits: {MAX_TOKENS_PER_DAY:,} tokens/day")
    print(f"[INFO] Output: {OUTPUT_JSONL}")
    if MAX_SAMPLES:
        print(f"[INFO] Limited to {MAX_SAMPLES} samples")
    print()
    
    # Load test data
    print("[INFO] Loading test data...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    if MAX_SAMPLES and MAX_SAMPLES < len(test_df):
        test_df = test_df.head(MAX_SAMPLES)
    
    print(f"[INFO] Total samples: {len(test_df)}")
    
    # Load existing predictions
    cache = load_predictions(OUTPUT_JSONL)
    print()
    
    # Generate predictions
    print("[INFO] Generating predictions...")
    processed = 0
    cached_count = len(cache)
    error_count = 0
    stopped_early = False
    
    for idx in tqdm(range(len(test_df)), desc="Processing", ncols=100):
        row = test_df.iloc[idx]
        img_id = str(row['img_id'])
        question = str(row['question'])
        gt_answer = str(row['answer'])
        
        # Check cache
        cache_key = (img_id, question)
        if cache_key in cache:
            continue
        
        # Load image
        img_path = os.path.join(IMAGE_FOLDER, f"{img_id}.jpg")
        image_base64 = encode_image_base64(img_path)
        
        # Query GPT-4o
        result = query_gpt4o(client, image_base64, question, rate_limiter)
        
        # Check if stopped due to rate limit
        if result['error'] in ['rate_limit_reached', 'api_rate_limit']:
            stopped_early = True
            print(f"\n[WARN] Stopped at sample {idx+1}/{len(test_df)}")
            break
        
        # Track errors
        if result['error']:
            error_count += 1
        
        # Save prediction
        record = {
            'img_id': img_id,
            'question': question,
            'ground_truth': gt_answer,
            'raw_output': result['raw_output'],
            'error': result['error'],
            'input_tokens': result['input_tokens'],
            'output_tokens': result['output_tokens'],
            'total_tokens': result['input_tokens'] + result['output_tokens']
        }
        save_prediction(OUTPUT_JSONL, record)
        processed += 1
        
        # Rate limiting delay
        time.sleep(RATE_LIMIT_DELAY)
    
    # Print statistics
    stats = rate_limiter.get_stats()
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Cached: {cached_count}")
    print(f"New: {processed}")
    print(f"Errors: {error_count}")
    print(f"Total: {cached_count + processed}/{len(test_df)}")
    print(f"-" * 70)
    print(f"API Usage:")
    print(f"  Requests: {stats['requests']}")
    print(f"  Input tokens: {stats['input_tokens']:,}")
    print(f"  Output tokens: {stats['output_tokens']:,}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Daily limit: {MAX_TOKENS_PER_DAY:,}")
    print(f"  Usage: {stats['total_tokens']/MAX_TOKENS_PER_DAY*100:.1f}%")
    print("="*70)
    
    if stopped_early:
        remaining = len(test_df) - (cached_count + processed)
        print(f"\n⚠️  Stopped early due to rate limits")
        print(f"   Remaining: {remaining} samples")
        print(f"   Run again tomorrow to continue!")
    else:
        print(f"\n✅ All predictions generated!")
    
    print(f"\n📁 Predictions saved to: {OUTPUT_JSONL}")
    print(f"   Use eval_gpt4o_results.py to evaluate")

if __name__ == "__main__":
    main()
