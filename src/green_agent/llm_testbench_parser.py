"""
LLM-based Testbench Output Parser

Since we're now fetching benchmarks dynamically from an external repository,
we can't standardize testbench outputs. This module uses LLM to intelligently
parse iverilog simulation output and determine test pass/fail status.
"""

import os
import re
from typing import Optional, Tuple, List
from pathlib import Path


class LLMTestbenchParser:
    """
    Uses LLM to parse testbench output and extract test results
    """
    
    def __init__(self):
        self.available = self._check_llm_available()
        self._client = None
        self._model = None
        
        if self.available:
            self._initialize_llm()
    
    def _check_llm_available(self) -> bool:
        """Check if LLM API keys are available"""
        return bool(
            os.environ.get("OPENAI_API_KEY") or 
            os.environ.get("ANTHROPIC_API_KEY")
        )
    
    def _initialize_llm(self):
        """Initialize LLM client"""
        try:
            # Try OpenAI first
            if os.environ.get("OPENAI_API_KEY"):
                from openai import OpenAI
                self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                self._model = "gpt-4o-mini"  # Fast and cheap
                self._provider = "openai"
            # Try Anthropic
            elif os.environ.get("ANTHROPIC_API_KEY"):
                from anthropic import Anthropic
                self._client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                self._model = "claude-3-haiku-20240307"  # Fast and cheap
                self._provider = "anthropic"
        except ImportError:
            self.available = False
            print("Warning: LLM libraries not installed. LLM-based parsing unavailable.")
    
    def parse_testbench_output(
        self, 
        simulation_log: str,
        task_id: str
    ) -> Tuple[int, int, int]:
        """
        Parse testbench output using LLM
        
        Args:
            simulation_log: Raw output from vvp simulation
            task_id: Task identifier for context
            
        Returns:
            Tuple of (passed, failed, total)
        """
        # First try rule-based parsing (fast, no cost)
        rule_based_result = self._try_rule_based_parsing(simulation_log)
        if rule_based_result is not None:
            return rule_based_result
        
        # Fall back to LLM parsing if available
        if self.available and self._client:
            return self._llm_parse(simulation_log, task_id)
        else:
            # Last resort: conservative parsing
            return self._conservative_parse(simulation_log)
    
    def _try_rule_based_parsing(self, output: str) -> Optional[Tuple[int, int, int]]:
        """
        Try to parse with rule-based patterns first (fast, free)
        
        Supports common patterns:
        - "Passed: X, Failed: Y" (our old standard)
        - "Tests passed: X/Y"
        - "PASS: X FAIL: Y"
        - "X tests passed, Y tests failed"
        """
        # Pattern 1: "Passed: X, Failed: Y"
        match = re.search(r'Passed:\s*(\d+),\s*Failed:\s*(\d+)', output, re.IGNORECASE)
        if match:
            passed = int(match.group(1))
            failed = int(match.group(2))
            return (passed, failed, passed + failed)
        
        # Pattern 2: "Tests passed: X/Y" or "X/Y tests passed"
        match = re.search(r'(\d+)/(\d+)\s*tests?\s*passed', output, re.IGNORECASE)
        if match:
            passed = int(match.group(1))
            total = int(match.group(2))
            return (passed, total - passed, total)
        
        # Pattern 3: "PASS: X FAIL: Y" or "PASS X FAIL Y"
        match = re.search(r'PASS:?\s*(\d+).*?FAIL:?\s*(\d+)', output, re.IGNORECASE | re.DOTALL)
        if match:
            passed = int(match.group(1))
            failed = int(match.group(2))
            return (passed, failed, passed + failed)
        
        # Pattern 4: "X tests passed, Y tests failed"
        match = re.search(r'(\d+)\s*tests?\s*passed.*?(\d+)\s*tests?\s*failed', output, re.IGNORECASE | re.DOTALL)
        if match:
            passed = int(match.group(1))
            failed = int(match.group(2))
            return (passed, failed, passed + failed)
        
        # Pattern 5: Count "Test X: PASS" and "Test X: FAIL"
        pass_matches = re.findall(r'Test\s*#?\d+:?\s*(?:PASS|passed|PASSED)', output, re.IGNORECASE)
        fail_matches = re.findall(r'Test\s*#?\d+:?\s*(?:FAIL|failed|FAILED)', output, re.IGNORECASE)
        
        if pass_matches or fail_matches:
            passed = len(pass_matches)
            failed = len(fail_matches)
            return (passed, failed, passed + failed)
        
        return None
    
    def _llm_parse(self, simulation_log: str, task_id: str) -> Tuple[int, int, int]:
        """
        Use LLM to intelligently parse simulation output
        """
        # Truncate log if too long (keep first and last parts)
        max_log_length = 8000
        if len(simulation_log) > max_log_length:
            truncated_log = (
                simulation_log[:max_log_length // 2] + 
                "\n\n... (truncated) ...\n\n" + 
                simulation_log[-max_log_length // 2:]
            )
        else:
            truncated_log = simulation_log
        
        prompt = f"""You are analyzing Verilog testbench simulation output for task: {task_id}

Parse the simulation log below and determine:
1. How many tests PASSED
2. How many tests FAILED
3. Total number of tests

Look for patterns like:
- "Test X: PASS" or "Test X: FAIL"
- "PASSED" or "FAILED" messages
- Error messages indicating failures
- Success messages indicating passes
- Test completion messages

SIMULATION LOG:
```
{truncated_log}
```

Respond in this EXACT format (just the numbers, no explanation):
PASSED: <number>
FAILED: <number>
TOTAL: <number>"""

        try:
            if self._provider == "openai":
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": "You are a precise test result parser. Always respond in the exact format requested."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=100
                )
                result_text = response.choices[0].message.content.strip()
            
            elif self._provider == "anthropic":
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=100,
                    temperature=0,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result_text = response.content[0].text.strip()
            
            # Parse LLM response
            passed = 0
            failed = 0
            total = 0
            
            for line in result_text.split('\n'):
                if 'PASSED:' in line:
                    passed = int(re.search(r'(\d+)', line).group(1))
                elif 'FAILED:' in line:
                    failed = int(re.search(r'(\d+)', line).group(1))
                elif 'TOTAL:' in line:
                    total = int(re.search(r'(\d+)', line).group(1))
            
            # Validate
            if total == 0 and (passed > 0 or failed > 0):
                total = passed + failed
            
            return (passed, failed, total)
            
        except Exception as e:
            print(f"Warning: LLM parsing failed: {e}")
            return self._conservative_parse(simulation_log)
    
    def _conservative_parse(self, output: str) -> Tuple[int, int, int]:
        """
        Conservative fallback: analyze output for obvious pass/fail indicators
        """
        # Look for error keywords
        error_keywords = [
            'error', 'fail', 'failed', 'incorrect', 'mismatch', 
            'wrong', 'invalid', 'assertion'
        ]
        
        success_keywords = [
            'pass', 'passed', 'success', 'correct', 'ok', 'all tests'
        ]
        
        output_lower = output.lower()
        
        # Count lines with errors vs success
        lines = output.split('\n')
        error_count = 0
        success_count = 0
        
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in error_keywords):
                error_count += 1
            elif any(kw in line_lower for kw in success_keywords):
                success_count += 1
        
        # If we found explicit test results, use them
        if error_count > 0 or success_count > 0:
            return (success_count, error_count, success_count + error_count)
        
        # Otherwise, assume failure if there's any output (testbenches usually print on failure)
        if len(output.strip()) > 50:  # More than trivial output
            return (0, 1, 1)  # Conservative: assume failure
        else:
            return (1, 0, 1)  # Minimal output: assume pass
    
    def is_available(self) -> bool:
        """Check if LLM parsing is available"""
        return self.available


def get_llm_parser() -> LLMTestbenchParser:
    """Factory function to get LLM parser instance"""
    return LLMTestbenchParser()
