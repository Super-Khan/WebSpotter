#!/usr/bin/env python3
"""
PKDD LLM Prompt Self-Refinement for Malicious Payload Detection
=============================================================

Uses PKDD dataset to let LLMs refine their own prompts through feedback,
handles multi-class labels and PKDD format directly.
"""

import json
import time
import asyncio
import os
from typing import List, Dict, Tuple
import csv
from datetime import datetime
import logging

# Simple imports
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

class PKDDPromptRefinement:
    def __init__(self):
        self.llms = {
            "claude-4-sonnet": {
                "type": "anthropic", 
                "model": "claude-sonnet-4-20250514",
                "input_cost": 3,
                "output_cost": 15,
                "key": "ANTHROPIC_API_KEY"
            },
            # "gpt-4o": {
            #     "type": "openai",
            #     "model": "gpt-4o",
            #     "input_cost": 5,
            #     "output_cost": 15,
            #     "key": "OPENAI_API_KEY"
            # },
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
        
        # Initialize detailed logging
        self.setup_logging()
        
        # Initial prompt template with attack type classification
        self.initial_prompt_template = """Please analyze the web request and determine if it contains malicious payloads. If it does, classify the attack type.

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

        # PKDD label mapping - matches actual dataset structure
        self.label_mapping = {
            0: "Valid",           
            1: "SqlInjection",    
            2: "XPathInjection",  
            3: "PathTransversal", 
            4: "OsCommanding",    
            5: "LdapInjection",   
            6: "SSI",           
            7: "XSS"              
        }
        
        # Reverse mapping for classification
        self.attack_type_to_label = {v: k for k, v in self.label_mapping.items()}

    def setup_logging(self):
        """Setup comprehensive logging for all LLM interactions"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger("PKDDPromptRefinement")
        self.logger.setLevel(logging.DEBUG)
        
        # Create file handler for detailed logs
        log_file = f"logs/pkdd_refinement_detailed_{timestamp}.log"
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
        self.logger.info("PKDD PROMPT REFINEMENT SESSION STARTED")
        self.logger.info("="*80)
        
    def log_llm_interaction(self, interaction_type: str, prompt: str, response: str, 
                           tokens: int, time_taken: float, model: str, metadata: dict = None):
        """Log detailed LLM interaction"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"LLM INTERACTION: {interaction_type}")
        self.logger.info(f"Model: {model}")
        self.logger.info(f"Tokens: {tokens}, Time: {time_taken:.2f}s")
        if metadata:
            self.logger.info(f"Metadata: {metadata}")
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

    def create_refinement_request(self, original_prompt: str, sample: Dict, 
                                llm_response: str, actual_attacks: List[str], 
                                feedback_type: str) -> str:
        """Create a prompt asking LLM to refine its detection prompt"""
        
        actual_attack_type = sample["type"]
        
        if feedback_type == "false_negative":
            feedback = f"""Your detection failed! You said it was safe, but there are actual {actual_attack_type} attacks: {actual_attacks}
You missed detecting these malicious payloads."""
            
        elif feedback_type == "false_positive":
            feedback = f"""Your detection failed! You found attacks but this request is actually safe (Valid).
Your prompt is too aggressive and finding false positives."""
            
        elif feedback_type == "wrong_attack_type":
            feedback = f"""Your detection failed! You detected the wrong attack type.
This is actually a {actual_attack_type} attack, not what you detected.
Actual attacks: {actual_attacks}
Your response: {llm_response}
"""
            
        elif feedback_type == "correct_type_wrong_location":
            feedback = f"""Your detection partially failed! You correctly identified it as {actual_attack_type} but found wrong locations.
Actual attack locations: {actual_attacks}
Your response: {llm_response}
"""
            
        else:
            feedback = "Your detection was incorrect. Please improve the prompt."

        refinement_request = f"""You are helping to improve malicious payload detection. Your current prompt failed on this test case:

SAMPLE:
Method: {sample["method"]}
URL: {sample["url"]}
Body: {sample["body"]}
Attack Type: {actual_attack_type}
Actual attacks: {actual_attacks if actual_attacks else "None (safe request)"}

YOUR CURRENT PROMPT:
{original_prompt}

YOUR RESPONSE WAS:
{llm_response}

FEEDBACK:
{feedback}

Please refine the prompt so it can better detect and classify malicious payloads, especially {actual_attack_type} attacks.

CRITICAL REQUIREMENTS for your refined prompt:

1. MUST include all these attack types:
   - Valid (normal/benign request)
   - XSS (Cross-Site Scripting)
   - SqlInjection (SQL Injection)
   - LdapInjection (LDAP Injection)
   - XPathInjection (XPATH Injection)
   - PathTransversal (Path Traversal)
   - OsCommanding (Command Execution)
   - SSI (SSI Attacks)

2. MUST require this EXACT output format with NO additional text:
   Attack Type: [one of the types above]
   Locations: [specific parameter/field locations if malicious, or "None" if Valid]
   
   The LLM must respond with ONLY these two lines and no explanation or analysis.

3. MUST contain these exact placeholders for input data:
   Method: {{method}}
   URL: {{url}}
   Body: {{body}}

These placeholders will be filled with actual request data when testing. Do NOT replace them with actual values.

Return ONLY the refined prompt, nothing else.

REFINED PROMPT:"""

        return refinement_request

    def validate_prompt_template(self, prompt: str) -> bool:
        """Check if prompt contains required placeholders and format requirements"""
        required_placeholders = ["{method}", "{url}", "{body}"]
        has_placeholders = all(placeholder in prompt for placeholder in required_placeholders)
        
        # Check for required format elements
        has_attack_type_format = "Attack Type:" in prompt
        has_locations_format = "Locations:" in prompt
        
        return has_placeholders and has_attack_type_format and has_locations_format

    def fix_prompt_template(self, prompt: str) -> str:
        """Fix prompt if it's missing placeholders or format requirements"""
        # First, comprehensively escape any template injection examples that might interfere with formatting
        
        # Step 1: Protect existing double braces
        prompt = prompt.replace('{{{{', '<<<QUAD_OPEN>>>').replace('}}}}', '<<<QUAD_CLOSE>>>')
        prompt = prompt.replace('{{', '<<<DOUBLE_OPEN>>>').replace('}}', '<<<DOUBLE_CLOSE>>>')
        
        # Step 2: Protect our required placeholders
        prompt = prompt.replace('{method}', '<<<METHOD>>>').replace('{url}', '<<<URL>>>').replace('{body}', '<<<BODY>>>')
        
        # Step 3: Escape any remaining single braces
        prompt = prompt.replace('{', '{{').replace('}', '}}')
        
        # Step 4: Restore our placeholders
        prompt = prompt.replace('<<<METHOD>>>', '{method}').replace('<<<URL>>>', '{url}').replace('<<<BODY>>>', '{body}')
        
        # Step 5: Restore the original brace patterns
        prompt = prompt.replace('<<<DOUBLE_OPEN>>>', '{{').replace('<<<DOUBLE_CLOSE>>>', '}}')
        prompt = prompt.replace('<<<QUAD_OPEN>>>', '{{{{').replace('<<<QUAD_CLOSE>>>', '}}}}')
        
        if self.validate_prompt_template(prompt):
            return prompt
        
        # If placeholders are missing, try to add them back
        if 'Method:' not in prompt and 'URL:' not in prompt:
            # Add the data section if completely missing
            lines = prompt.split('\n')
            # Insert before "Format your response" or at the end
            insert_pos = len(lines)
            for i, line in enumerate(lines):
                if 'format your response' in line.lower() or 'response:' in line.lower():
                    insert_pos = i
                    break
            
            data_section = '''
HTTP Request:
Method: {method}
URL: {url}
Body: {body}
'''
            lines.insert(insert_pos, data_section)
            prompt = '\n'.join(lines)
        
        # Add format requirements if missing
        if "Attack Type:" not in prompt or "Locations:" not in prompt:
            if "IMPORTANT:" not in prompt:
                prompt += '''

IMPORTANT: You must respond EXACTLY in this format and nothing else:

Attack Type: [one of the types above]
Locations: [specific parameter/field locations if malicious, or "None" if Valid]

Do not provide any explanation, analysis, or additional text. Only respond with the two lines above.'''
        
        return prompt

    async def call_anthropic(self, model: str, prompt: str, max_tokens: int = 1000) -> tuple:
        """Call Anthropic API"""
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        start = time.time()
        
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
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
            model=model
        )
        
        return content, elapsed, tokens

    async def call_xai(self, model: str, prompt: str, max_tokens: int = 1000) -> tuple:
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
            model=model
        )
        
        return content, elapsed, tokens

    def parse_response(self, response: str) -> tuple:
        """Parse response to extract attack type and locations (matches benchmark.py)"""
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
        """Compare predicted attack locations with actual attacks (same as benchmark.py)"""
        if not actual_attacks and not predicted_locations:
            return True  # Both found no attacks
        
        if not actual_attacks and predicted_locations:
            return False  # False positive
        
        if actual_attacks and not predicted_locations:
            return False  # False negative
        
        # Check if any predicted location matches any actual attack
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

    async def test_prompt_on_sample(self, prompt_template: str, sample: Dict, llm_config: Dict) -> Dict:
        """Test a prompt on a single PKDD sample"""
        # Fill in the prompt template - escape any unmatched braces first
        try:
            prompt = prompt_template.format(
                method=sample["method"],
                url=sample["url"], 
                body=sample["body"]
            )
        except (ValueError, IndexError) as e:
            # If formatting fails due to unmatched braces, escape them and try again
            self.logger.warning(f"Prompt formatting failed, escaping braces: {e}")
            
            # More comprehensive escaping - handle all problematic brace patterns
            escaped_template = prompt_template
            
            # First escape existing double braces to prevent interference
            escaped_template = escaped_template.replace('{{{{', '<<<QUAD_OPEN>>>').replace('}}}}', '<<<QUAD_CLOSE>>>')
            escaped_template = escaped_template.replace('{{', '<<<DOUBLE_OPEN>>>').replace('}}', '<<<DOUBLE_CLOSE>>>')
            
            # Now escape any remaining single braces that aren't our placeholders
            import re
            # Protect our placeholders first
            escaped_template = escaped_template.replace('{method}', '<<<METHOD>>>').replace('{url}', '<<<URL>>>').replace('{body}', '<<<BODY>>>')
            
            # Escape any remaining single braces
            escaped_template = escaped_template.replace('{', '{{').replace('}', '}}')
            
            # Restore our placeholders
            escaped_template = escaped_template.replace('<<<METHOD>>>', '{method}').replace('<<<URL>>>', '{url}').replace('<<<BODY>>>', '{body}')
            
            # Restore the original brace patterns
            escaped_template = escaped_template.replace('<<<DOUBLE_OPEN>>>', '{{').replace('<<<DOUBLE_CLOSE>>>', '}}')
            escaped_template = escaped_template.replace('<<<QUAD_OPEN>>>', '{{{{').replace('<<<QUAD_CLOSE>>>', '}}}}')
            
            prompt = escaped_template.format(
                method=sample["method"],
                url=sample["url"], 
                body=sample["body"]
            )
        
        # Log sample testing start
        attack_type = self.label_mapping.get(sample["label"], f"Unknown({sample['label']})")
        self.logger.info(f"\n{'*'*50}")
        self.logger.info(f"TESTING SAMPLE: {sample['method']} {sample['url'][:50]}...")
        self.logger.info(f"Attack Type: {attack_type}")
        self.logger.info(f"Actual Label: {sample['label']}")
        self.logger.info(f"Actual Attacks: {sample.get('attacks', [])}")
        self.logger.info(f"{'*'*50}")
        
        # Get LLM response based on provider type
        if llm_config["type"] == "anthropic":
            response, elapsed, tokens = await self.call_anthropic(llm_config["model"], prompt)
        elif llm_config["type"] == "openai":
            base_url = llm_config.get("base_url")  # For Grok
            response, elapsed, tokens = await self.call_openai(llm_config["model"], prompt, base_url=base_url)
        elif llm_config["type"] == "xai":
            response, elapsed, tokens = await self.call_xai(llm_config["model"], prompt)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_config['type']}")
        
        # Parse response - updated to handle attack type classification
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
        
        print(f"Testing sample: {sample['method']} {sample['url'][:50]}... - Type: {actual_attack_type}, Predicted: {predicted_attack_type}")
        print(f"Attack type correct: {attack_type_correct}, Location correct: {location_correct}, Overall: {is_correct}")
        print(f"Actual attacks: {actual_attacks}, Predicted locations: {predicted_locations}")

        # Evaluate correctness and determine feedback type
        if actual_label == 0:
            if predicted_label == 0:
                feedback_type = "correct"
            else:
                feedback_type = "false_positive"
        else:
            if predicted_label == 0:
                feedback_type = "false_negative"
            elif attack_type_correct and location_correct:
                feedback_type = "correct"
            elif attack_type_correct and not location_correct:
                feedback_type = "correct_type_wrong_location"
            else:
                feedback_type = "wrong_attack_type"
        
        # Log evaluation results
        self.logger.info(f"EVALUATION RESULT:")
        self.logger.info(f"  Actual Attack Type: {actual_attack_type}")
        self.logger.info(f"  Predicted Attack Type: {predicted_attack_type}")
        self.logger.info(f"  Attack Type Correct: {attack_type_correct}")
        self.logger.info(f"  Predicted Locations: {predicted_locations}")
        self.logger.info(f"  Location Correct: {location_correct}")
        self.logger.info(f"  Overall Correct: {is_correct}")
        self.logger.info(f"  Feedback Type: {feedback_type}")
        self.logger.info("")
        
        return {
            "is_correct": is_correct,
            "feedback_type": feedback_type,
            "predicted_label": predicted_label,
            "predicted_attack_type": predicted_attack_type,
            "predicted_locations": predicted_locations,
            "actual_attacks": actual_attacks,
            "llm_response": response,
            "tokens": tokens,
            "time": elapsed,
            "attack_type_correct": attack_type_correct,
            "location_correct": location_correct
        }

    async def refine_prompt_on_sample(self, sample: Dict, llm_name: str, current_prompt: str, max_rounds: int = 3) -> Dict:
        """Refine prompt on a single PKDD sample through multiple rounds"""
        config = self.llms[llm_name]
        
        attack_type = self.label_mapping.get(sample["label"], f"Unknown({sample['label']})")
        
        self.logger.info(f"\n{'#'*70}")
        self.logger.info(f"STARTING PROMPT REFINEMENT ON SAMPLE")
        self.logger.info(f"Sample: {sample['method']} {sample['url'][:50]}...")
        self.logger.info(f"Attack Type: {attack_type}, Label: {sample['label']}, Attacks: {sample.get('attacks', [])}")
        self.logger.info(f"{'#'*70}")
        
        history = []
        
        for round_num in range(1, max_rounds + 1):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"REFINEMENT ROUND {round_num}")
            self.logger.info(f"{'='*50}")
            self.logger.info(f"Current prompt length: {len(current_prompt)} chars")
            
            print(f"    Round {round_num}: Testing current prompt...")
            
            # Test current prompt
            result = await self.test_prompt_on_sample(current_prompt, sample, config)
            
            history.append({
                "round": round_num,
                "prompt": current_prompt,
                "result": result
            })
            
            if result["is_correct"]:
                self.logger.info(f"SUCCESS in round {round_num}!")
                print(f"    Success in round {round_num}!")
                return {
                    "success": True,
                    "final_prompt": current_prompt,
                    "rounds_needed": round_num,
                    "history": history
                }
            
            if round_num == max_rounds:
                self.logger.info(f"FAILED after {max_rounds} rounds")
                print(f"    Failed after {max_rounds} rounds")
                break
                
            # Generate refinement request
            self.logger.info(f"Failed with {result['feedback_type']}, generating refinement request...")
            print(f"    Failed with {result['feedback_type']}, refining prompt...")
            
            refinement_request = self.create_refinement_request(
                current_prompt, sample, result["llm_response"], 
                result["actual_attacks"], result["feedback_type"]
            )
            
            # Log refinement request
            self.logger.info(f"REFINEMENT REQUEST GENERATED:")
            self.logger.info("-" * 40)
            self.logger.info(refinement_request)
            self.logger.info("-" * 40)
            
            # Get refined prompt based on provider type
            if config["type"] == "anthropic":
                refined_prompt, _, _ = await self.call_anthropic(config["model"], refinement_request)
            elif config["type"] == "openai":
                base_url = config.get("base_url")  # For Grok
                refined_prompt, _, _ = await self.call_openai(config["model"], refinement_request, base_url=base_url)
            else:
                raise ValueError(f"Unsupported LLM type: {config['type']}")
            
            current_prompt = refined_prompt.strip()
            
            # Fix any formatting issues in the refined prompt
            current_prompt = self.fix_prompt_template(current_prompt)
            
            self.logger.info(f"NEW REFINED PROMPT (Length: {len(current_prompt)} chars):")
            self.logger.info("-" * 40)
            self.logger.info(current_prompt)
            self.logger.info("-" * 40)
            
            await asyncio.sleep(0.5)  # Rate limiting
        
        return {
            "success": False,
            "final_prompt": current_prompt,
            "rounds_needed": max_rounds,
            "history": history
        }

    def load_pkdd_dataset(self, filename: str) -> List[Dict]:
        """Load PKDD dataset"""
        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    async def run_pkdd_prompt_refinement(self, train_samples: int = 20, test_samples: int = 30, llm_name: str = None):
        """Main refinement process for PKDD dataset"""
        print("PKDD LLM Prompt Self-Refinement for Malicious Payload Detection")
        print(f"Training on {train_samples} samples, testing on {test_samples} samples...")
        
        # Load PKDD datasets from separate train and test files
        print("Loading PKDD training data...")
        train_full_data = self.load_pkdd_dataset("PKDD_train.jsonl")
        print("Loading PKDD test data...")
        test_full_data = self.load_pkdd_dataset("PKDD_test.jsonl")
        
        # Take requested number of samples
        train_data = train_full_data[:train_samples]
        test_data = test_full_data[:test_samples]
        
        print(f"Using {len(train_data)} training samples from PKDD_train.jsonl")
        print(f"Using {len(test_data)} test samples from PKDD_test.jsonl")
        
        # Select LLM to use
        if llm_name and llm_name in self.llms:
            selected_llm = llm_name
        else:
            # Show available LLMs and let user choose
            print("\nAvailable LLMs:")
            for i, name in enumerate(self.llms.keys(), 1):
                print(f"  {i}. {name}")
            
            if llm_name:
                print(f"Warning: '{llm_name}' not found, using default.")
            
            selected_llm = list(self.llms.keys())[0]  # Default to first
        
        print(f"Using {selected_llm} for refinement...")
        
        # Phase 1: Refinement on training data
        print("\n" + "="*60)
        print("PHASE 1: PROMPT REFINEMENT ON PKDD TRAINING DATA")
        print("="*60)
        
        refinement_results = []
        # Start with initial prompt and evolve it through training
        evolving_prompt = self.initial_prompt_template
        
        for i, sample in enumerate(train_data):
            attack_type = self.label_mapping.get(sample["label"], f"Unknown({sample['label']})")
            print(f"\nSample {i+1}/{len(train_data)}: {sample['method']} - {attack_type}")
            print(f"Starting with prompt length: {len(evolving_prompt)} chars")
            
            result = await self.refine_prompt_on_sample(sample, selected_llm, evolving_prompt)
            refinement_results.append({
                "sample_id": i+1,
                "sample": sample,
                "result": result,
                "starting_prompt": evolving_prompt
            })
            
            # Update evolving prompt with the refined version
            evolving_prompt = result["final_prompt"]
            
            if result["success"]:
                print(f"  → Success! Rounds needed: {result['rounds_needed']}")
            else:
                print(f"  → Failed after 3 rounds, but keeping refined prompt")
            
            print(f"  → Updated prompt length: {len(evolving_prompt)} chars")
        
        # Analyze refinement results
        final_prompt = evolving_prompt
        success_rate = sum(1 for r in refinement_results if r["result"]["success"]) / len(refinement_results)
        print(f"\nRefinement Success Rate: {success_rate:.2%}")
        print(f"Using final evolved prompt for testing (length: {len(final_prompt)} chars)")
        
        # Phase 2: Test refined prompt
        print("\n" + "="*60)
        print("PHASE 2: TESTING REFINED PROMPT ON PKDD DATA")
        print("="*60)
        
        test_results = await self.test_refined_prompt(final_prompt, test_data, selected_llm)
        
        # Save results
        self.save_pkdd_results(refinement_results, test_results, final_prompt)
        
        # Save prompt lengths
        self.save_prompt_lengths(refinement_results)
        
        print(f"\nFinal Test Accuracy: {test_results['accuracy']:.3f}")
        print(f"Results saved to pkdd_refinement_results.csv, pkdd_test_results.csv, and prompt_lengths.csv")

    async def test_refined_prompt(self, prompt_template: str, test_data: List[Dict], llm_name: str) -> Dict:
        """Test the refined prompt on PKDD test data"""
        config = self.llms[llm_name]
        
        results = []
        correct = 0
        total_time = 0
        total_tokens = 0
        
        for i, sample in enumerate(test_data):
            attack_type = self.label_mapping.get(sample["label"], f"Unknown({sample['label']})")
            print(f"Testing sample {i+1}/{len(test_data)} ({attack_type})")
            
            result = await self.test_prompt_on_sample(prompt_template, sample, config)
            
            if result["is_correct"]:
                correct += 1
            
            total_time += result["time"]
            total_tokens += result["tokens"]
            
            results.append({
                "sample_id": i+1,
                "sample": sample,
                "result": result
            })
            
            await asyncio.sleep(0.1)
        
        accuracy = correct / len(test_data)
        avg_time = total_time / len(test_data)
        cost = total_tokens * config["input_cost"] / 1000000
        
        return {
            "accuracy": accuracy,
            "avg_time": avg_time,
            "total_cost": cost,
            "total_tokens": total_tokens,
            "results": results
        }

    def save_pkdd_results(self, refinement_results: List[Dict], test_results: Dict, final_prompt: str):
        """Save PKDD refinement results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save refinement process
        with open(f"pkdd_refinement_results_{timestamp}.csv", "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Sample_ID", "Method", "URL", "Body", "Attack_Type", "Label", "Actual_Attacks", 
                           "Success", "Rounds_Needed", "Starting_Prompt_Length", "Final_Prompt_Length", 
                           "Final_Response", "Final_Predicted"])
            
            for r in refinement_results:
                sample = r["sample"]
                result = r["result"]
                last_result = result["history"][-1]["result"]
                attack_type = self.label_mapping.get(sample["label"], f"Unknown({sample['label']})")
                
                writer.writerow([
                    r["sample_id"],
                    sample["method"],
                    sample["url"][:100] + "..." if len(sample["url"]) > 100 else sample["url"],
                    sample["body"][:200] + "..." if len(sample["body"]) > 200 else sample["body"],
                    attack_type,
                    sample["label"],
                    "; ".join(sample.get("attacks", [])) if sample.get("attacks") else "None",
                    result["success"],
                    result["rounds_needed"],
                    len(r["starting_prompt"]),
                    len(result["final_prompt"]),
                    last_result["llm_response"][:300] + "..." if len(last_result["llm_response"]) > 300 else last_result["llm_response"],
                    "; ".join(last_result["predicted_locations"]) if last_result["predicted_locations"] else "None"
                ])
        
        # Save test results - following benchmark detailed_results format with attack type info
        with open(f"pkdd_test_results_{timestamp}.csv", "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["LLM", "Sample_ID", "Method", "URL", "Body", 
                           "Actual_Attack_Type", "Predicted_Attack_Type", "Attack_Type_Correct",
                           "Actual_Label", "Predicted_Label", 
                           "Actual_Attacks", "Predicted_Locations", "Location_Correct",
                           "Overall_Correct", "Match_Type", "LLM_Response", "Time_sec", "Tokens"])
            
            llm_name = list(self.llms.keys())[0]  # Get the LLM name used for testing
            
            for r in test_results["results"]:
                sample = r["sample"]
                result = r["result"]
                
                # Map feedback_type to match_type for consistency with benchmark
                if sample["label"] == 0:
                    if result["predicted_label"] == 0:
                        match_type = "True Negative"
                    else:
                        match_type = "False Positive"
                else:
                    if result["predicted_label"] == 0:
                        match_type = "False Negative"
                    elif result["attack_type_correct"] and result["location_correct"]:
                        match_type = "True Positive"
                    elif result["attack_type_correct"] and not result["location_correct"]:
                        match_type = "Correct Type, Wrong Location"
                    else:
                        match_type = "Wrong Attack Type"
                
                writer.writerow([
                    llm_name,
                    r["sample_id"],
                    sample["method"],
                    sample["url"][:100] + "..." if len(sample["url"]) > 100 else sample["url"],
                    sample["body"][:200] + "..." if len(sample["body"]) > 200 else sample["body"],
                    sample["type"],  # Actual attack type
                    result["predicted_attack_type"],
                    result["attack_type_correct"],
                    sample["label"],  # Actual label
                    result["predicted_label"],
                    "; ".join(sample.get("attacks", [])) if sample.get("attacks") else "None",
                    "; ".join(result["predicted_locations"]) if result["predicted_locations"] else "None",
                    result["location_correct"],
                    result["is_correct"],
                    match_type,
                    result["llm_response"].replace('\n', ' ').replace('\r', ' ')[:500] + "..." if len(result["llm_response"]) > 500 else result["llm_response"].replace('\n', ' ').replace('\r', ' '),
                    f"{result['time']:.2f}",
                    result["tokens"]
                ])
        
        # Save final prompt
        with open(f"pkdd_final_prompt_{timestamp}.txt", "w", encoding='utf-8') as f:
            f.write("FINAL REFINED PROMPT (PKDD):\n")
            f.write("="*50 + "\n")
            f.write(final_prompt)
            f.write("\n\n")
            f.write(f"Test Accuracy: {test_results['accuracy']:.3f}\n")
            f.write(f"Test Cost: ${test_results['total_cost']:.4f}\n")

    def save_prompt_lengths(self, refinement_results: List[Dict]):
        """Save final prompt length for each training sample"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(f"prompt_lengths_{timestamp}.csv", "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Sample_ID", "Attack_Type", "Final_Prompt_Length", "Success", "Rounds_Needed"])
            
            for r in refinement_results:
                sample = r["sample"]
                result = r["result"]
                attack_type = self.label_mapping.get(sample["label"], f"Unknown({sample['label']})")
                final_prompt_length = len(result["final_prompt"])
                
                writer.writerow([
                    r["sample_id"],
                    attack_type,
                    final_prompt_length,
                    result["success"],
                    result["rounds_needed"]
                ])

async def main():
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass
    
    refinement = PKDDPromptRefinement()
        
    await refinement.run_pkdd_prompt_refinement(train_samples=10, test_samples=5)

if __name__ == "__main__":
    asyncio.run(main())
