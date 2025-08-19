#!/usr/bin/env python3
"""
Evaluates GPT-4o-mini on PKDD dataset using MSU-based payload detection.
"""

import json
import time
import asyncio
import openai
import os
import re
from typing import List, Dict, Any
import csv
from datetime import datetime
from urllib.parse import urlparse

from reproduce import RequestInfo, get_http_level_split_furl

try:
    import openai
except ImportError:
    print("Error: openai package not found. Install with: pip install openai")
    exit(1)

class PKDDMSUEvaluator:
    def __init__(self):
        self.model = "gpt-4o-mini"
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.prompt_template = """Definition: Your task is to locate malicious payloads in an HTTP request. The HTTP request will be divided into minimal semantic units (MSUs), and the input is an array of strings, where each string represents an MSU of the HTTP request. The output should be a dictionary in JSON format, where the key is a string from the input array, and the value is 0 or 1. A value of 0 indicates that the unit does not contain malicious payloads, and a value of 1 means otherwise.

Example-1:
Input: ["POST", "/tienda1", "/<marquee loop=1 width=0 onfinish=alert(1)>", "/anadir.jsp", "id=2", "nombre=Iber", "precio=5003", "cantidad=64", "B1=Entrar"]
Output: {"POST": 0, "/tienda1": 0, "/<marquee loop=1 width=0 onfinish=alert(1)>": 1, "/anadir.jsp": 0, "id=2": 0, "nombre=Iber": 0, "precio=5003": 0, "cantidad=64": 0, "B1=Entrar": 0}

User Input: INPUT_PLACEHOLDER

Respond with ONLY the JSON dictionary, no additional text or explanation."""

    def load_pkdd_dataset(self, filename: str) -> List[Dict]:
        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def pkdd_to_request_info(self, sample: Dict) -> RequestInfo:
        return RequestInfo(
            method=sample["method"],
            url=sample["url"],
            body=sample.get("body", "")
        )

    def extract_attack_locations_from_msu(self, sample: Dict, msu_list: List[str]) -> List[str]:
        """Extract which MSUs should contain attacks based on PKDD annotations."""
        if sample["label"] == 0:
            return []
        
        attacks = sample.get("attacks", [])
        malicious_msus = []
        
        if attacks:
            for msu in msu_list:
                for attack in attacks:
                    if self.msu_contains_attack(msu, attack):
                        malicious_msus.append(msu)
                        break
        
        return malicious_msus

    def msu_contains_attack(self, msu: str, attack: str) -> bool:
        msu_norm = msu.lower().replace(' ', '').replace('+', ' ')
        attack_norm = attack.lower().replace(' ', '').replace('+', ' ')
        
        if attack_norm in msu_norm or msu_norm in attack_norm:
            return True
        
        if '=' in msu and '=' in attack:
            msu_param = msu.split('=')[0].strip()
            attack_param = attack.split('=')[0].strip()
            if msu_param.lower() == attack_param.lower():
                return True
        
        return False

    async def call_gpt4o_mini(self, msu_list: List[str]) -> Dict[str, int]:
        input_str = json.dumps(msu_list)
        prompt = self.prompt_template.replace("INPUT_PLACEHOLDER", input_str)
        
        print(f"  JSON Input: {input_str}")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0
            )
            
            response_text = response.choices[0].message.content.strip()
            print(f"  Raw LLM Response: {response_text}")

            
            # Try to parse JSON response
            try:
                
                # Clean response text and remove any markdown formatting
                cleaned_text = response_text
                result = json.loads(cleaned_text)
                
                # Ensure all MSUs are in the result and convert to proper format
                final_result = {}
                for i, msu in enumerate(msu_list):
                    if msu in result:
                        # Ensure value is 0 or 1
                        val = result[msu]
                        final_result[msu] = 1 if val in [1, "1", True] else 0
                    else:
                        final_result[msu] = 0  # Default to non-malicious
                
                print(f"  Final Result: {final_result}")
                return final_result
                
            except json.JSONDecodeError as je:
                print(f"  JSON PARSING FAILED: {je}")
                print(f"  Returning all zeros...")
                # Return all MSUs as non-malicious if parsing fails
                return {msu: 0 for msu in msu_list}
                
        except Exception as e:
            print(f"  API CALL FAILED: {e}")
            print(f"  Returning all zeros...")
            # Return all MSUs as non-malicious if API fails
            return {msu: 0 for msu in msu_list}

    def evaluate_prediction(self, predicted: Dict[str, int], expected_malicious: List[str], 
                           is_actually_malicious: bool) -> Dict[str, Any]:
        predicted_malicious_msus = [msu for msu, val in predicted.items() if val == 1]
        has_malicious_prediction = len(predicted_malicious_msus) > 0
        
        if is_actually_malicious:
            if expected_malicious:
                is_correct = any(pred_msu in expected_malicious for pred_msu in predicted_malicious_msus)
            else:
                is_correct = has_malicious_prediction
        else:
            is_correct = not has_malicious_prediction
        
        return {
            "is_correct": is_correct,
            "has_malicious_prediction": has_malicious_prediction,
            "predicted_malicious_msus": predicted_malicious_msus
        }

    async def evaluate_sample(self, sample: Dict) -> Dict[str, Any]:
        req_info = self.pkdd_to_request_info(sample)
        msu_list = get_http_level_split_furl(req_info)
        
        expected_malicious = self.extract_attack_locations_from_msu(sample, msu_list)
        predicted = await self.call_gpt4o_mini(msu_list)
        
        is_actually_malicious = sample["label"] > 0
        
        evaluation = self.evaluate_prediction(predicted, expected_malicious, is_actually_malicious)
        
        return {
            "sample_id": sample.get("id", "unknown"),
            "attack_type": sample.get("type", "Unknown"),
            "msu_list": msu_list,
            "expected_malicious": expected_malicious,
            "predicted_malicious": evaluation["predicted_malicious_msus"],
            "is_correct": evaluation["is_correct"],
            "is_actually_malicious": is_actually_malicious,
            "has_malicious_prediction": evaluation["has_malicious_prediction"]
        }

    async def run_evaluation(self, dataset_file: str = "PKDD_test.jsonl", max_samples: int = 100):
        print(f"PKDD MSU Evaluation with GPT-4o-mini")
        print(f"Loading dataset: {dataset_file}")
        
        data = self.load_pkdd_dataset(dataset_file)
        if max_samples:
            data = data[:max_samples]
        
        print(f"Evaluating {len(data)} samples...")
        
        results = []
        correct_count = 0
        
        for i, sample in enumerate(data):
            print(f"Processing sample {i+1}/{len(data)}: {sample['method']} {sample['url'][:50]}...")
            
            try:
                result = await self.evaluate_sample(sample)
                results.append(result)
                
                if result["is_correct"]:
                    correct_count += 1
                
                print(f"  Result: {'CORRECT' if result['is_correct'] else 'INCORRECT'}")
                print(f"  Actual Attack: {sample.get('attacks', []) if sample['label'] > 0 else 'None (benign)'}")
                if result['predicted_malicious']:
                    print(f"  Predicted: {result['predicted_malicious']}")
                if result['expected_malicious']:
                    print(f"  Expected: {result['expected_malicious']}")
                print()  # Empty line for spacing
                
            except Exception as e:
                print(f"  Error processing sample: {e}")
                continue
            
            # Rate limiting
            await asyncio.sleep(0.1)
        
        # Calculate overall accuracy
        total_samples = len(results)
        accuracy = correct_count / total_samples if total_samples > 0 else 0
        
        # Save results
        self.save_results(results, accuracy)
        
        # Print summary
        print("=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Samples: {total_samples}")
        print(f"Correct Predictions: {correct_count}")
        print(f"Overall Accuracy: {accuracy:.3f}")
        print("=" * 60)
        
        return accuracy

    def save_results(self, results: List[Dict], accuracy: float):
        """Save simplified evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        with open(f"pkdd_msu_results_{timestamp}.csv", "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Sample_ID", "Attack_Type", "Method", "URL", 
                "Is_Actually_Malicious", "Has_Malicious_Prediction", "Is_Correct",
                "Expected_Malicious", "Predicted_Malicious", "MSU_List"
            ])
            
            for result in results:
                writer.writerow([
                    result["sample_id"],
                    result["attack_type"],
                    result["msu_list"][0] if result["msu_list"] else "",  # Method
                    result["msu_list"][1] if len(result["msu_list"]) > 1 else "",  # URL
                    result["is_actually_malicious"],
                    result["has_malicious_prediction"], 
                    result["is_correct"],
                    "; ".join(result["expected_malicious"]),
                    "; ".join(result["predicted_malicious"]),
                    "; ".join(result["msu_list"])
                ])
        
        # Save summary
        with open(f"pkdd_msu_summary_{timestamp}.txt", "w", encoding='utf-8') as f:
            f.write("PKDD MSU Evaluation Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Total Samples: {len(results)}\n")
            f.write(f"Overall Accuracy: {accuracy:.3f}\n\n")
            
            correct_count = sum(1 for r in results if r["is_correct"])
            f.write(f"Correct Predictions: {correct_count}\n")
            f.write(f"Incorrect Predictions: {len(results) - correct_count}\n\n")
            
            malicious_count = sum(1 for r in results if r["is_actually_malicious"])
            benign_count = len(results) - malicious_count
            f.write(f"Malicious Samples: {malicious_count}\n")
            f.write(f"Benign Samples: {benign_count}\n")

async def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    evaluator = PKDDMSUEvaluator()
    
    await evaluator.run_evaluation(
        dataset_file="PKDD_test.jsonl",
        max_samples=10 
    )

if __name__ == "__main__":
    asyncio.run(main())
