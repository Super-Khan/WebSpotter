#!/usr/bin/env python3
"""
PKDD LLM Malicious Payload Detection Benchmark
=============================================

Tests LLMs on PKDD malicious payload detection dataset.
Measures: accuracy, response time, token cost.
Handles multi-class labels and PKDD format directly.
"""

import json
import time
import asyncio
import os
from typing import List, Dict
import csv
import logging
from datetime import datetime

# Simple imports - install only what you need
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from xai_sdk import AsyncClient as XAIAsyncClient
except ImportError:
    XAIAsyncClient = None

try:
    import httpx
except ImportError:
    httpx = None

class PKDDBenchmark:
    def __init__(self):
        # Initialize detailed logging
        self.setup_logging()
        
        # Only available LLMs with real pricing
        self.llms = {
            # "claude-4-sonnet": {
            #     "type": "anthropic", 
            #     "model": "claude-sonnet-4-20250514",
            #     "input_cost": 3,
            #     "output_cost": 15,
            #     "key": "ANTHROPIC_API_KEY"
            # },
            "gpt-4o": {
                "type": "openai",
                "model": "chatgpt-4o-latest",
                "input_cost": 5,
                "output_cost": 15,
                "key": "OPENAI_API_KEY"
            },
            # "gpt-4o-mini": {
            #     "type": "openai",
            #     "model": "gpt-4o-mini",
            #     "input_cost": 0.15,
            #     "output_cost": 0.6,
            #     "key": "OPENAI_API_KEY"
            # },
            # "gemini-2.5-pro": {
            #     "type": "google",
            #     "model": "gemini-2.5-pro",
            #     "input_cost": 1.25,
            #     "output_cost": 10,
            #     "key": "GOOGLE_AI_API_KEY"
            # },
            # "grok-4": {
            #     "type": "openai",  
            #     "model": "grok-4-0709",
            #     "input_cost": 3,
            #     "output_cost": 15,
            #     "key": "XAI_API_KEY",
            #     "base_url": "https://api.x.ai/v1"
            # }
        }
        
        # PKDD label mapping - matches actual dataset structure
        self.label_mapping = {
            0: "Valid",           # Normal Query
            1: "SqlInjection",    # SQL Injection
            2: "XPathInjection",  # XPATH Injection
            3: "PathTransversal", # Path Traversal
            4: "OsCommanding",    # Command Execution
            5: "LdapInjection",   # LDAP Injection
            6: "SSI",             # SSI Attacks
            7: "XSS"              # Cross-Site Scripting
        }
        
        # Reverse mapping for classification
        self.attack_type_to_label = {v: k for k, v in self.label_mapping.items()}
        
    def setup_logging(self):
        """Setup comprehensive logging for all LLM interactions"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger("PKDDBenchmark")
        self.logger.setLevel(logging.DEBUG)
        
        # Create file handler for detailed logs
        log_file = f"logs/pkdd_benchmark_detailed_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Log session start
        self.logger.info("="*80)
        self.logger.info("PKDD BENCHMARK SESSION STARTED")
        self.logger.info("="*80)
        
    def log_llm_interaction(self, interaction_type: str, prompt: str, response: str, 
                           tokens: int, time_taken: float, model: str, sample_info: dict = None):
        """Log detailed LLM interaction"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"LLM INTERACTION: {interaction_type}")
        self.logger.info(f"Model: {model}")
        self.logger.info(f"Tokens: {tokens}, Time: {time_taken:.2f}s")
        if sample_info:
            self.logger.info(f"Sample Info: {sample_info}")
        self.logger.info(f"{'='*60}")
        
        self.logger.info("PROMPT SENT:")
        self.logger.info("-" * 40)
        self.logger.info(prompt)
        self.logger.info("-" * 40)
        
        self.logger.info("RESPONSE RECEIVED:")
        self.logger.info("-" * 40)
        self.logger.info(response)
        self.logger.info("-" * 40)
        self.logger.info("")
        
    def create_prompt(self, method: str, url: str, body: str) -> str:
        """Create prompt for PKDD dataset with attack type classification"""
        return f"""Please analyze the web request and determine if it contains malicious payloads. If it does, classify the attack type.

HTTP Request:
Method: {method}
URL: {url}
Body: {body}

Analyze this request and determine:
1. Is it malicious or benign?
2. If malicious, what type of attack is it?

Available attack types:
- Valid (normal/benign request)
- XSS (Cross-Site Scripting)
- SqlInjection (SQL Injection)
- LdapInjection (LDAP Injection)
- XPathInjection (XPATH Injection)
- PathTransversal (Path Traversal)
- OsCommanding (Command Execution)
- SSI (SSI Attacks)

If there are malicious payloads, also specify the exact locations where they appear (e.g., parameter names, URL paths, etc.).

IMPORTANT: You must respond EXACTLY in this format and nothing else:

Attack Type: [one of the types above]
Locations: [specific parameter/field locations if malicious, or "None" if Valid]

Do not provide any explanation, analysis, or additional text. Only respond with the two lines above."""

    def parse_response(self, response: str) -> tuple:
        """Parse response to extract attack type and locations"""
        response = response.strip()
        
        # Initialize defaults
        predicted_attack_type = "Valid"
        attack_locations = []
        
        # Extract attack type
        import re
        
        # Look for "Attack Type:" pattern (more flexible search)
        attack_type_match = re.search(r'Attack Type:\s*([^\n\r]+)', response, re.IGNORECASE)
        if attack_type_match:
            predicted_attack_type = attack_type_match.group(1).strip()
        
        # Clean up attack type name (remove extra spaces, punctuation)
        predicted_attack_type = re.sub(r'[^\w\s]', '', predicted_attack_type).strip()
        
        # Normalize attack type to match our mapping (more comprehensive)
        attack_type_lower = predicted_attack_type.lower()
        if any(word in attack_type_lower for word in ['valid', 'benign', 'normal', 'safe']):
            predicted_attack_type = "Valid"
        elif any(word in attack_type_lower for word in ['xss', 'cross-site', 'script', 'cross site']):
            predicted_attack_type = "XSS"
        elif any(word in attack_type_lower for word in ['sql', 'sqlinjection']):
            predicted_attack_type = "SqlInjection"
        elif any(word in attack_type_lower for word in ['ldap', 'ldapinjection']):
            predicted_attack_type = "LdapInjection"
        elif any(word in attack_type_lower for word in ['xpath', 'xml', 'xpathinjection']):
            predicted_attack_type = "XPathInjection"
        elif any(word in attack_type_lower for word in ['path', 'traversal', 'directory', 'pathtraversal']):
            predicted_attack_type = "PathTransversal"
        elif any(word in attack_type_lower for word in ['command', 'execution', 'os', 'oscommanding']):
            predicted_attack_type = "OsCommanding"
        elif any(word in attack_type_lower for word in ['ssi', 'server-side', 'server side']):
            predicted_attack_type = "SSI"
        
        # Extract locations (look for "Locations:" pattern)
        locations_match = re.search(r'Locations:\s*([^\n\r]+)', response, re.IGNORECASE)
        if locations_match:
            locations_text = locations_match.group(1).strip()
            if locations_text.lower() not in ['none', 'n/a', 'not applicable', 'nil']:
                # Split by common delimiters and clean up
                attack_locations = [loc.strip() for loc in re.split(r'[,;]', locations_text) if loc.strip()]
        
        # If no structured format found, fall back to parsing the full response
        if not attack_type_match:
            response_lower = response.lower()
            
            # Check for explicit statements
            if any(phrase in response_lower for phrase in ["safe", "valid", "benign", "no malicious", "no payload", "not malicious"]):
                predicted_attack_type = "Valid"
            else:
                # Try to guess from content keywords
                if any(word in response_lower for word in ['script', 'xss', 'alert', 'javascript', '<script']):
                    predicted_attack_type = "XSS"
                elif any(word in response_lower for word in ['sql', 'union', 'select', 'injection', 'or 1=1']):
                    predicted_attack_type = "SqlInjection"
                elif any(word in response_lower for word in ['../', 'path', 'traversal', 'directory', '../']):
                    predicted_attack_type = "PathTransversal"
                elif any(word in response_lower for word in ['xpath', 'xml', 'node()']):
                    predicted_attack_type = "XPathInjection"
                elif any(word in response_lower for word in ['ldap']):
                    predicted_attack_type = "LdapInjection"
                elif any(word in response_lower for word in ['command', 'cmd', 'exec', 'system']):
                    predicted_attack_type = "OsCommanding"
                elif any(word in response_lower for word in ['ssi', 'server-side']):
                    predicted_attack_type = "SSI"
                else:
                    predicted_attack_type = "Valid"  # Default to safe if unsure
            
            # Try to extract locations from the response text if no structured format
            if not locations_match and predicted_attack_type != "Valid":
                # Look for parameter=value patterns or quoted strings
                param_patterns = re.findall(r'([a-zA-Z0-9_]+)=([^,\s\n]+)', response)
                for param, value in param_patterns:
                    attack_locations.append(f"{param}={value}")
                
                # Look for quoted parameter mentions
                quoted_patterns = re.findall(r'"([^"]*=.*?)"', response)
                for pattern in quoted_patterns:
                    attack_locations.append(pattern)
        
        # Convert attack type to label number
        predicted_label = self.attack_type_to_label.get(predicted_attack_type, 0)
        
        return predicted_label, predicted_attack_type, attack_locations

    def compare_attack_locations(self, predicted_locations: list, actual_attacks: list) -> bool:
        """Compare predicted attack locations with actual attacks"""
        if not actual_attacks and not predicted_locations:
            return True  # Both found no attacks
        
        if not actual_attacks and predicted_locations:
            return False  # False positive
        
        if actual_attacks and not predicted_locations:
            return False  # False negative
        
        # Check if any predicted location matches any actual attack
        # We'll use fuzzy matching since formats might differ slightly
        for predicted in predicted_locations:
            for actual in actual_attacks:
                # Normalize both strings for comparison
                pred_norm = predicted.lower().replace(' ', '').replace('+', ' ')
                actual_norm = actual.lower().replace(' ', '').replace('+', ' ')
                
                # Check if they match or if one contains the other
                if pred_norm in actual_norm or actual_norm in pred_norm:
                    return True
                
                # Check parameter name matching (before =)
                if '=' in predicted and '=' in actual:
                    pred_param = predicted.split('=')[0].strip()
                    actual_param = actual.split('=')[0].strip()
                    if pred_param.lower() == actual_param.lower():
                        return True
        
        return False

    async def call_anthropic(self, model: str, prompt: str, sample_info: dict = None) -> tuple:
        """Call Anthropic API"""
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        start = time.time()
        
        response = await client.messages.create(
            model=model,
            max_tokens=200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        elapsed = time.time() - start
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        # Log the interaction
        self.log_llm_interaction(
            interaction_type="ANTHROPIC_API_CALL",
            prompt=prompt,
            response=content,
            tokens=tokens,
            time_taken=elapsed,
            model=model,
            sample_info=sample_info
        )
        
        return content, elapsed, tokens

    async def call_openai(self, model: str, prompt: str, sample_info: dict = None, max_tokens: int = 1000, base_url: str = None) -> tuple:
        """Call OpenAI API (also handles Grok)"""
        if base_url:
            # For Grok or other OpenAI-compatible APIs
            client = openai.AsyncOpenAI(
                api_key=os.getenv("XAI_API_KEY"),
                base_url="https://api.x.ai/v1"
            )
        else:
            # Standard OpenAI
            client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        start = time.time()
        
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0
        )
        
        elapsed = time.time() - start
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens
        
        # Log the interaction
        api_type = "GROK_API_CALL" if base_url else "OPENAI_API_CALL"
        self.log_llm_interaction(
            interaction_type=api_type,
            prompt=prompt,
            response=content,
            tokens=tokens,
            time_taken=elapsed,
            model=model,
            sample_info=sample_info
        )
        
        return content, elapsed, tokens

    async def call_google(self, model: str, prompt: str, sample_info: dict = None,max_tokens: int = 1000) -> tuple:
        """Call Google AI"""
        genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))
        model_instance = genai.GenerativeModel(model)
        start = time.time()
        
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            temperature=0,
            max_output_tokens=max_tokens
        )
        
        # Configure safety settings to be less restrictive for security analysis
        safety_settings = [
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_NONE
            }
        ]
        
        try:
            response = await model_instance.generate_content_async(
                prompt, 
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            elapsed = time.time() - start
            
            # Check if response was blocked
            if response.candidates and response.candidates[0].finish_reason != genai.types.FinishReason.STOP:
                # Response was blocked, return a default safe response
                content = "Attack Type: Valid\nLocations: None"
                print(f"Warning: Google API blocked response (finish_reason: {response.candidates[0].finish_reason}), using default")
            else:
                content = response.text
                
        except Exception as e:
            elapsed = time.time() - start
            # If there's any error, return a default safe response
            content = "Attack Type: Valid\nLocations: None"
            print(f"Warning: Google API error ({e}), using default response")
        
        # Rough token estimate for Google
        tokens = len(prompt.split()) + len(content.split())
        
        # Log the interaction
        self.log_llm_interaction(
            interaction_type="GOOGLE_AI_API_CALL",
            prompt=prompt,
            response=content,
            tokens=tokens,
            time_taken=elapsed,
            model=model,
            sample_info=sample_info
        )
        
        return content, elapsed, tokens

    async def call_xai(self, model: str, prompt: str, sample_info: dict = None, max_tokens: int = 1000) -> tuple:
        """Call XAI (Grok) API using official XAI SDK"""
        if not XAIAsyncClient:
            raise ImportError("XAI SDK not available. Install with: pip install xai-sdk")
        
        client = XAIAsyncClient(api_key=os.getenv("XAI_API_KEY"))
        start = time.time()
        
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0
            )
            
            elapsed = time.time() - start
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if hasattr(response, 'usage') else len(prompt.split()) + len(content.split())
            
        except Exception as e:
            elapsed = time.time() - start
            # If there's any error, return a default safe response
            content = "Attack Type: Valid\nLocations: None"
            tokens = len(prompt.split()) + len(content.split())
            print(f"Warning: XAI API error ({e}), using default response")
        
        # Log the interaction
        self.log_llm_interaction(
            interaction_type="XAI_API_CALL",
            prompt=prompt,
            response=content,
            tokens=tokens,
            time_taken=elapsed,
            model=model,
            sample_info=sample_info
        )
        
        return content, elapsed, tokens

    async def test_llm(self, llm_name: str, samples: List[Dict], max_samples: int = 50) -> Dict:
        """Test one LLM on PKDD dataset"""
        config = self.llms[llm_name]
        
        # Check API key
        if not os.getenv(config["key"]):
            print(f"ERROR: {llm_name}: No API key")
            return {}
            
        print(f"Testing {llm_name} on PKDD dataset...")
        
        results = []
        total_time = 0
        total_tokens = 0
        correct = 0
        
        test_samples = samples[:max_samples]
        
        for i, sample in enumerate(test_samples):
            try:
                prompt = self.create_prompt(sample["method"], sample["url"], sample["body"])
                
                # Prepare sample info for logging
                sample_info = {
                    "sample_id": i+1,
                    "method": sample["method"],
                    "url": sample["url"][:50] + "..." if len(sample["url"]) > 50 else sample["url"],
                    "actual_label": sample["label"],
                    "attack_type": sample["type"],
                    "actual_attacks": sample.get("attacks", [])
                }
                
                # Call API based on provider type
                if config["type"] == "anthropic":
                    response, elapsed, tokens = await self.call_anthropic(config["model"], prompt, sample_info)
                elif config["type"] == "openai":
                    base_url = config.get("base_url")
                    response, elapsed, tokens = await self.call_openai(config["model"], prompt, sample_info, base_url=base_url)
                elif config["type"] == "google":
                    response, elapsed, tokens = await self.call_google(config["model"], prompt, sample_info)
                elif config["type"] == "xai":
                    response, elapsed, tokens = await self.call_xai(config["model"], prompt, sample_info)
                else:
                    raise ValueError(f"Unsupported LLM type: {config['type']}")
                
                predicted_label, predicted_attack_type, predicted_locations = self.parse_response(response)
                actual_label = sample["label"]
                actual_attack_type = sample["type"]
                actual_attacks = sample.get("attacks", [])
                
                # Evaluate correctness based on attack type classification
                attack_type_correct = predicted_attack_type == actual_attack_type
                
                # For location matching, only check if both agree it's malicious
                location_correct = True
                if actual_label > 0 and predicted_label > 0:
                    location_correct = self.compare_attack_locations(predicted_locations, actual_attacks)
                
                # Overall correctness: both attack type and location must be correct
                is_correct = attack_type_correct and (actual_label == 0 or location_correct)
                
                # Determine match type for detailed analysis
                if actual_label == 0:
                    if predicted_label == 0:
                        match_type = "True Negative"
                    else:
                        match_type = "False Positive"
                else:
                    if predicted_label == 0:
                        match_type = "False Negative"
                    elif attack_type_correct and location_correct:
                        match_type = "True Positive"
                    elif attack_type_correct and not location_correct:
                        match_type = "Correct Type, Wrong Location"
                    else:
                        match_type = "Wrong Attack Type"
                
                if is_correct:
                    correct += 1
                    
                total_time += elapsed
                total_tokens += tokens
                
                # Log evaluation results
                self.logger.info(f"EVALUATION RESULT for Sample {i+1}:")
                self.logger.info(f"  Actual Attack Type: {actual_attack_type} (label: {actual_label})")
                self.logger.info(f"  Predicted Attack Type: {predicted_attack_type} (label: {predicted_label})")
                self.logger.info(f"  Attack Type Correct: {attack_type_correct}")
                self.logger.info(f"  Predicted Locations: {predicted_locations}")
                self.logger.info(f"  Actual Attacks: {actual_attacks}")
                self.logger.info(f"  Location Correct: {location_correct}")
                self.logger.info(f"  Match Type: {match_type}")
                self.logger.info(f"  Overall Correct: {is_correct}")
                self.logger.info("")
                
                results.append({
                    "sample_id": i+1,
                    "actual_label": actual_label,
                    "actual_attack_type": actual_attack_type,
                    "predicted_label": predicted_label,
                    "predicted_attack_type": predicted_attack_type,
                    "attack_type_correct": attack_type_correct,
                    "actual_attacks": actual_attacks,
                    "predicted_locations": predicted_locations,
                    "location_correct": location_correct,
                    "correct": is_correct,
                    "match_type": match_type,
                    "time": elapsed,
                    "tokens": tokens,
                    "llm_response": response,
                    "method": sample["method"],
                    "url": sample["url"],
                    "body": sample["body"]
                })
                
                if (i + 1) % 10 == 0:
                    print(f"   Completed {i+1}/{len(test_samples)}")
                    
                await asyncio.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"   Error on sample {i+1}: {e}")
                continue
        
        # Calculate metrics
        accuracy = correct / len(test_samples)
        avg_time = total_time / len(test_samples)
        cost = total_tokens * config["input_cost"] / 1000000 
        
        # Calculate attack type specific metrics
        attack_type_metrics = {}
        for attack_type in self.label_mapping.values():
            type_results = [r for r in results if r["actual_attack_type"] == attack_type]
            if type_results:
                type_correct = sum(1 for r in type_results if r["attack_type_correct"])
                attack_type_metrics[attack_type] = {
                    "count": len(type_results),
                    "correct": type_correct,
                    "accuracy": type_correct / len(type_results)
                }

        return {
            "accuracy": accuracy,
            "avg_time": avg_time,
            "total_cost": cost,
            "total_tokens": total_tokens,
            "samples_tested": len(test_samples),
            "attack_type_metrics": attack_type_metrics,
            "results": results
        }

    def load_pkdd_dataset(self, filename: str) -> List[Dict]:
        """Load PKDD JSONL dataset"""
        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def save_results(self, all_results: Dict):
        """Save PKDD benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary results
        with open(f"pkdd_benchmark_results_{timestamp}.csv", "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["LLM", "Accuracy", "Avg_Time_sec", "Total_Cost_USD", "Total_Tokens"])
            
            for llm, metrics in all_results.items():
                if metrics:
                    writer.writerow([
                        llm,
                        f"{metrics['accuracy']:.3f}",
                        f"{metrics['avg_time']:.2f}",
                        f"{metrics['total_cost']:.4f}",
                        metrics['total_tokens']
                    ])
        
        # Save detailed results with PKDD-specific fields including attack type classification
        with open(f"pkdd_detailed_results_{timestamp}.csv", "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["LLM", "Sample_ID", "Method", "URL", "Body", 
                           "Actual_Attack_Type", "Predicted_Attack_Type", "Attack_Type_Correct",
                           "Actual_Label", "Predicted_Label", 
                           "Actual_Attacks", "Predicted_Locations", "Location_Correct",
                           "Overall_Correct", "Match_Type", "LLM_Response", "Time_sec", "Tokens"])
            
            for llm, metrics in all_results.items():
                if metrics and "results" in metrics:
                    for result in metrics["results"]:
                        writer.writerow([
                            llm,
                            result["sample_id"],
                            result["method"],
                            result["url"][:100] + "..." if len(result["url"]) > 100 else result["url"],
                            result["body"][:200] + "..." if len(result["body"]) > 200 else result["body"],
                            result["actual_attack_type"],
                            result["predicted_attack_type"],
                            result["attack_type_correct"],
                            result["actual_label"],
                            result["predicted_label"],
                            "; ".join(result["actual_attacks"]) if result["actual_attacks"] else "None",
                            "; ".join(result["predicted_locations"]) if result["predicted_locations"] else "None",
                            result["location_correct"],
                            result["correct"],
                            result["match_type"],
                            result["llm_response"].replace('\n', ' ').replace('\r', ' ')[:500] + "..." if len(result["llm_response"]) > 500 else result["llm_response"].replace('\n', ' ').replace('\r', ' '),
                            f"{result['time']:.2f}",
                            result["tokens"]
                        ])

    async def run_benchmark(self, max_samples: int = 50):
        """Run the PKDD benchmark"""
        print("PKDD LLM Benchmark for Malicious Payload Detection")
        print(f"Testing with {max_samples} samples...")
        
        # Load PKDD data
        dataset = self.load_pkdd_dataset("PKDD_test.jsonl")
        print(f"Loaded {len(dataset)} total PKDD samples")
        
        # Analyze dataset
        label_counts = {}
        for sample in dataset:
            label = sample["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("\nPKDD Label Distribution:")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            attack_type = self.label_mapping.get(label, f"Unknown({label})")
            print(f"  {label} ({attack_type}): {count} samples")
        
        # Test each LLM
        results = {}
        for llm_name in self.llms.keys():
            try:
                metrics = await self.test_llm(llm_name, dataset, max_samples)
                results[llm_name] = metrics
            except Exception as e:
                print(f"ERROR: {llm_name} failed: {e}")
                results[llm_name] = {}
        
        # Show results
        print("\n" + "="*60)
        print("PKDD BENCHMARK RESULTS")
        print("="*60)
        
        for llm, metrics in results.items():
            if metrics:
                print(f"\n{llm.upper()}")
                print(f"   Overall Accuracy: {metrics['accuracy']:.3f}")
                print(f"   Avg Time:        {metrics['avg_time']:.2f}s")
                print(f"   Cost:            ${metrics['total_cost']:.4f}")
                print(f"   Tokens:          {metrics['total_tokens']:,}")
                
                # Show attack type specific accuracies
                if "attack_type_metrics" in metrics:
                    print(f"   Attack Type Accuracies:")
                    for attack_type, type_metrics in metrics["attack_type_metrics"].items():
                        print(f"     {attack_type:15}: {type_metrics['accuracy']:.3f} ({type_metrics['correct']}/{type_metrics['count']})")
            else:
                print(f"\nFAILED: {llm.upper()}")
        
        # Save results
        self.save_results(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\nResults saved to pkdd_benchmark_results_{timestamp}.csv")
        print(f"Detailed results saved to pkdd_detailed_results_{timestamp}.csv")

async def main():
    # Load environment variables if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass
    
    benchmark = PKDDBenchmark()
    await benchmark.run_benchmark(max_samples=500) 

if __name__ == "__main__":
    asyncio.run(main())
