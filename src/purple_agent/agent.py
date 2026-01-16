"""
ArchXBench Purple Agent - Baseline LLM-driven Verilog Generator

This agent solves RTL synthesis tasks by:
1. Querying the green agent for task details
2. Using an LLM to generate Verilog code
3. Submitting for evaluation
4. Iteratively refining based on feedback
"""

import os
import re
import json
import time
import requests
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod


@dataclass
class GenerationResult:
    """Result of attempting to solve a task"""
    task_id: str
    verilog_code: str
    success: bool
    iterations: int
    passed: int = 0
    failed: int = 0
    total: int = 0
    error_message: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Aggregated results for a benchmark run"""
    total_tasks: int
    passed: int
    failed: int
    pass_rate: float
    weighted_score: float
    results: List[GenerationResult]
    execution_time_sec: float


# ========== LLM Backend Abstraction ==========

class LLMBackend(ABC):
    """Abstract base class for LLM backends"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response from the LLM"""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI API backend"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        import openai
        openai.api_key = self.api_key
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=4000
        )
        
        return response.choices[0].message.content


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        
        message = client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system_prompt if system_prompt else "You are an expert Verilog RTL designer.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text


class LocalLLMBackend(LLMBackend):
    """Local LLM backend (Ollama, vLLM, etc.)"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "codellama"):
        self.base_url = base_url.rstrip("/")
        self.model = model
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]


class GeminiBackend(LLMBackend):
    """Google Gemini API backend"""
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)

        final_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        resp = model.generate_content(
            final_prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 4096,
            },
            safety_settings={
                # Keep defaults minimal; rely on prompt for safety
                "HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                "HATE": "BLOCK_MEDIUM_AND_ABOVE",
                "SEXUAL": "BLOCK_MEDIUM_AND_ABOVE",
                "DANGEROUS": "BLOCK_MEDIUM_AND_ABOVE",
            },
        )
        return resp.text


class OpenRouterBackend(LLMBackend):
    """OpenRouter (OpenAI-compatible) backend"""
    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-sonnet"):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=4000,
        )
        return resp.choices[0].message.content


# ========== Purple Agent Implementation ==========

class ArchXBenchPurpleAgent:
    """
    Purple Agent: LLM-based Verilog RTL Generator
    
    This agent demonstrates the A2A protocol for interacting with
    the ArchXBench green agent to solve RTL synthesis tasks.
    """
    
    SYSTEM_PROMPT = """You are an expert Verilog RTL designer with deep knowledge of:
- Digital logic design and synthesis
- SystemVerilog 2012 constructs
- Combinational and sequential circuits
- Arithmetic units, FIFOs, pipelines
- Best practices for synthesizable RTL

Always generate clean, synthesizable Verilog code that follows the exact
module signature specified in the design specs. Use proper coding style
with clear signal names and comments."""

    def __init__(
        self, 
        green_agent_url: str,
        llm_backend: LLMBackend,
        max_iterations: int = 5,
        verbose: bool = True,
        use_llm_validation: bool = True,
        generate_feedback: bool = True
    ):
        """
        Initialize the purple agent.
        
        Args:
            green_agent_url: URL of the green agent server
            llm_backend: LLM backend for code generation
            max_iterations: Maximum fix attempts per task
            verbose: Print progress messages
            use_llm_validation: Enable LLM-based architectural validation (recommended)
            generate_feedback: Generate detailed feedback for iterative improvement
        """
        self.green_agent_url = green_agent_url.rstrip("/")
        self.llm = llm_backend
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.use_llm_validation = use_llm_validation
        self.generate_feedback = generate_feedback
        self.session_id: Optional[str] = None
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    # ========== A2A Protocol: Green Agent Interaction ==========
    
    def get_available_tasks(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query green agent for available tasks"""
        params = {"level": level} if level else {}
        response = requests.get(f"{self.green_agent_url}/tasks", params=params)
        response.raise_for_status()
        return response.json()
    
    def get_task(self, task_id: str) -> Dict[str, Any]:
        """Get task details from green agent"""
        response = requests.get(f"{self.green_agent_url}/tasks/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def submit_solution(self, task_id: str, verilog_code: str) -> Dict[str, Any]:
        """Submit solution to green agent for evaluation"""
        payload = {
            "task_id": task_id,
            "verilog_code": verilog_code,
            "validate_architecture": True,
            "use_llm_validation": self.use_llm_validation,
            "generate_feedback": self.generate_feedback
        }
        if self.session_id:
            payload["session_id"] = self.session_id
        
        response = requests.post(
            f"{self.green_agent_url}/evaluate",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def create_session(self, agent_id: str) -> str:
        """Create evaluation session with green agent"""
        response = requests.post(
            f"{self.green_agent_url}/sessions",
            json={"agent_id": agent_id}
        )
        response.raise_for_status()
        self.session_id = response.json()["session_id"]
        return self.session_id
    
    def get_session_results(self) -> Dict[str, Any]:
        """Get session results from green agent"""
        if not self.session_id:
            raise ValueError("No active session")
        response = requests.get(f"{self.green_agent_url}/sessions/{self.session_id}")
        response.raise_for_status()
        return response.json()
    
    # ========== Code Generation ==========
    
    def solve_task(self, task_id: str) -> GenerationResult:
        """
        Solve a single task with iterative refinement.
        
        1. Generate initial Verilog from problem description
        2. Submit to green agent for evaluation
        3. If failed, use error feedback to fix
        4. Repeat until pass or max iterations
        """
        self._log(f"\n{'='*60}")
        self._log(f"Solving: {task_id}")
        self._log(f"{'='*60}")
        
        task = self.get_task(task_id)
        
        # Initial generation
        self._log("Generating initial solution...")
        verilog_code = self._generate_verilog(task)
        
        for iteration in range(1, self.max_iterations + 1):
            self._log(f"\nIteration {iteration}/{self.max_iterations}")
            
            # Submit and check
            result = self.submit_solution(task_id, verilog_code)
            
            self._log(f"  Compilation: {'✓' if result['compilation_success'] else '✗'}")
            self._log(f"  Tests: {result['passed']}/{result['total']} passed")
            
            # Log failed tests if any (handle missing/None error_message)
            if result['failed'] > 0:
                err_msg = result.get('error_message')
                if err_msg:
                    self._log(f"  Failed tests: {err_msg[:200]}...")
            
            if result["success"]:
                self._log(f"  ✓ PASSED!")
                return GenerationResult(
                    task_id=task_id,
                    verilog_code=verilog_code,
                    success=True,
                    iterations=iteration,
                    passed=result["passed"],
                    failed=result["failed"],
                    total=result["total"]
                )
            
            # Failed - try to fix if not last iteration or if compilation failed
            if iteration < self.max_iterations or not result['compilation_success']:
                self._log(f"  Refining solution...")
                verilog_code = self._fix_verilog(task, verilog_code, result)
        
        self._log(f"  ✗ FAILED after {self.max_iterations} iterations")
        return GenerationResult(
            task_id=task_id,
            verilog_code=verilog_code,
            success=False,
            iterations=self.max_iterations,
            passed=result.get("passed", 0),
            failed=result.get("failed", 0),
            total=result.get("total", 0),
            error_message=result.get("error_message")
        )
    
    def _generate_verilog(self, task: Dict[str, Any]) -> str:
        """Generate initial Verilog code using LLM"""
        prompt = f"""Generate synthesizable Verilog code for the following RTL design task.

## Problem Description
{task['problem_description']}

## Design Specifications
{task['design_specs']}

## Requirements
1. Follow the EXACT module signature from the design specs
2. Use only synthesizable Verilog constructs
3. Implement the complete functionality as described
4. Include appropriate comments
5. Handle all edge cases properly

## Output Format
Provide ONLY the Verilog module code wrapped in ```verilog ... ``` tags.
Do NOT include any testbench code.
"""
        response = self.llm.generate(prompt, self.SYSTEM_PROMPT)
        return self._extract_verilog(response)
    
    def _fix_verilog(
        self, 
        task: Dict[str, Any], 
        current_code: str, 
        error_result: Dict[str, Any]
    ) -> str:
        """Fix Verilog code based on error feedback"""
        # Build comprehensive error context
        error_context = f"""## Error Information
- Compilation Success: {error_result.get('compilation_success', True)}
- Tests Passed: {error_result.get('passed', 0)}
- Tests Failed: {error_result.get('failed', 0)}
- Error Message: {error_result.get('error_message', 'N/A')}
"""
        
        # Add detailed feedback if available
        if error_result.get('feedback'):
            feedback = error_result['feedback']
            
            if feedback.get('compilation_errors'):
                error_context += "\n## Compilation Errors\n"
                for err in feedback['compilation_errors']:
                    error_context += f"- {err}\n"
            
            if feedback.get('test_failures'):
                error_context += "\n## Test Failures\n"
                for test in feedback['test_failures']:
                    error_context += f"- Test #{test.get('test_number', '?')}: {test.get('description', 'Failed')}\n"
                    if test.get('expected'):
                        error_context += f"  Expected: {test['expected']}\n"
                    if test.get('actual'):
                        error_context += f"  Actual: {test['actual']}\n"
            
            if feedback.get('behavioral_issues'):
                error_context += "\n## Behavioral Issues\n"
                for issue in feedback['behavioral_issues']:
                    error_context += f"- {issue}\n"
            
            if feedback.get('suggestions'):
                error_context += "\n## Suggestions\n"
                for sugg in feedback['suggestions']:
                    error_context += f"- {sugg}\n"
        
        # Add architectural compliance feedback if available
        if error_result.get('architectural_compliance'):
            compliance = error_result['architectural_compliance']
            if compliance.get('violations'):
                error_context += "\n## Architectural Violations\n"
                for violation in compliance['violations']:
                    error_context += f"- {violation}\n"
            if compliance.get('warnings'):
                error_context += "\n## Architectural Warnings\n"
                for warning in compliance['warnings']:
                    error_context += f"- {warning}\n"
        
        prompt = f"""Fix the following Verilog code that failed testing.

## Problem Description
{task['problem_description']}

## Design Specifications
{task['design_specs']}

## Current (Failing) Code
```verilog
{current_code}
```

{error_context}

## Instructions
1. Carefully analyze the detailed feedback above
2. Identify the specific bug or issue causing the failures
3. Fix the implementation while maintaining the exact module signature
4. Address all compilation errors, test failures, and architectural violations
5. Ensure all edge cases are handled

## Output Format
Provide ONLY the corrected Verilog code wrapped in ```verilog ... ``` tags.
"""
        response = self.llm.generate(prompt, self.SYSTEM_PROMPT)
        return self._extract_verilog(response)
    
    def _extract_verilog(self, text: str) -> str:
        """Extract Verilog code from LLM response"""
        # Try to find code blocks
        patterns = [
            r'```verilog\n(.*?)```',
            r'```systemverilog\n(.*?)```',
            r'```v\n(.*?)```',
            r'```\n(.*?)```',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: look for module...endmodule
        match = re.search(r'(module\s+\w+.*?endmodule)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Last resort: return as-is
        return text.strip()
    
    # ========== Benchmark Execution ==========
    
    def run_benchmark(
        self, 
        levels: Optional[List[str]] = None,
        task_ids: Optional[List[str]] = None
    ) -> BenchmarkResult:
        """
        Run the full benchmark or a subset of tasks.
        
        Args:
            levels: Filter to specific levels (e.g., ["level-0", "level-1a"])
            task_ids: Specific tasks to run (overrides levels filter)
            
        Returns:
            BenchmarkResult with aggregated metrics
        """
        start_time = time.time()
        
        # Get tasks to run
        if task_ids:
            tasks = [{"task_id": tid} for tid in task_ids]
        else:
            tasks = self.get_available_tasks()
            if levels:
                tasks = [t for t in tasks if t["level"] in levels]
        
        self._log(f"\n{'#'*60}")
        self._log(f"# ArchXBench Purple Agent - Benchmark Run")
        self._log(f"# Tasks: {len(tasks)}")
        self._log(f"{'#'*60}")
        
        results = []
        for i, task_info in enumerate(tasks, 1):
            task_id = task_info["task_id"]
            self._log(f"\n[{i}/{len(tasks)}] ", )
            result = self.solve_task(task_id)
            results.append(result)
        
        execution_time = time.time() - start_time
        
        # Aggregate results
        total = len(results)
        passed = sum(1 for r in results if r.success)
        pass_rate = passed / total if total > 0 else 0
        
        # Simple weighted score (by task difficulty would need task info)
        weighted_score = pass_rate  # Simplified
        
        self._log(f"\n{'#'*60}")
        self._log(f"# BENCHMARK COMPLETE")
        self._log(f"# Pass Rate: {pass_rate*100:.1f}% ({passed}/{total})")
        self._log(f"# Time: {execution_time:.1f}s")
        self._log(f"{'#'*60}")
        
        return BenchmarkResult(
            total_tasks=total,
            passed=passed,
            failed=total - passed,
            pass_rate=pass_rate,
            weighted_score=weighted_score,
            results=results,
            execution_time_sec=execution_time
        )


# ========== Factory Functions ==========

def create_openai_agent(
    green_agent_url: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    **kwargs
) -> ArchXBenchPurpleAgent:
    """Create purple agent with OpenAI backend"""
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required")
    
    backend = OpenAIBackend(api_key, model)
    return ArchXBenchPurpleAgent(green_agent_url, backend, **kwargs)


def create_anthropic_agent(
    green_agent_url: str,
    api_key: Optional[str] = None,
    model: str = "claude-3-sonnet-20240229",
    **kwargs
) -> ArchXBenchPurpleAgent:
    """Create purple agent with Anthropic Claude backend"""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key required")
    
    backend = AnthropicBackend(api_key, model)
    return ArchXBenchPurpleAgent(green_agent_url, backend, **kwargs)


def create_local_agent(
    green_agent_url: str,
    llm_url: str = "http://localhost:11434",
    model: str = "codellama",
    **kwargs
) -> ArchXBenchPurpleAgent:
    """Create purple agent with local LLM backend (Ollama)"""
    backend = LocalLLMBackend(llm_url, model)
    return ArchXBenchPurpleAgent(green_agent_url, backend, **kwargs)


def create_gemini_agent(
    green_agent_url: str,
    api_key: Optional[str] = None,
    model: str = "gemini-1.5-pro",
    **kwargs
) -> ArchXBenchPurpleAgent:
    """Create purple agent with Gemini backend"""
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key required")

    backend = GeminiBackend(api_key, model)
    return ArchXBenchPurpleAgent(green_agent_url, backend, **kwargs)


def create_openrouter_agent(
    green_agent_url: str,
    api_key: Optional[str] = None,
    model: str = "anthropic/claude-3.5-sonnet",
    **kwargs
) -> ArchXBenchPurpleAgent:
    """Create purple agent with OpenRouter backend"""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API key required")

    backend = OpenRouterBackend(api_key, model)
    return ArchXBenchPurpleAgent(green_agent_url, backend, **kwargs)


# ========== CLI Entry Point ==========

def main():
    """Command-line interface for the purple agent"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ArchXBench Purple Agent - Baseline LLM Verilog Generator"
    )
    parser.add_argument(
        "--green-url", 
        default=os.environ.get("GREEN_AGENT_URL", "http://localhost:8000"),
        help="URL of the green agent server"
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "anthropic", "gemini", "openrouter", "local"],
        default="openai",
        help="LLM backend to use"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (defaults: gpt-4, claude-3-sonnet, codellama)"
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        help="Specific levels to run (e.g., level-0 level-1a)"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Specific task IDs to run"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum fix iterations per task"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--no-llm-validation",
        action="store_true",
        help="Disable LLM-based architectural validation (faster, less comprehensive)"
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Disable detailed feedback generation (faster, less informative)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Create agent with selected backend
    model_defaults = {
        "openai": "gpt-4",
        "anthropic": "claude-3-sonnet-20240229",
        "gemini": "gemini-1.5-pro",
        "openrouter": "anthropic/claude-3.5-sonnet",
        "local": "codellama"
    }
    model = args.model or model_defaults[args.backend]
    
    if args.backend == "openai":
        agent = create_openai_agent(
            args.green_url, 
            model=model,
            max_iterations=args.max_iterations,
            verbose=not args.quiet,
            use_llm_validation=not args.no_llm_validation,
            generate_feedback=not args.no_feedback
        )
    elif args.backend == "anthropic":
        agent = create_anthropic_agent(
            args.green_url,
            model=model,
            max_iterations=args.max_iterations,
            verbose=not args.quiet,
            use_llm_validation=not args.no_llm_validation,
            generate_feedback=not args.no_feedback
        )
    elif args.backend == "gemini":
        agent = create_gemini_agent(
            args.green_url,
            model=model,
            max_iterations=args.max_iterations,
            verbose=not args.quiet,
            use_llm_validation=not args.no_llm_validation,
            generate_feedback=not args.no_feedback
        )
    elif args.backend == "openrouter":
        agent = create_openrouter_agent(
            args.green_url,
            model=model,
            max_iterations=args.max_iterations,
            verbose=not args.quiet,
            use_llm_validation=not args.no_llm_validation,
            generate_feedback=not args.no_feedback
        )
    else:
        agent = create_local_agent(
            args.green_url,
            model=model,
            max_iterations=args.max_iterations,
            verbose=not args.quiet,
            use_llm_validation=not args.no_llm_validation,
            generate_feedback=not args.no_feedback
        )
    
    # Run benchmark
    results = agent.run_benchmark(levels=args.levels, task_ids=args.tasks)
    
    # Save results if requested
    if args.output:
        output_data = {
            "total_tasks": results.total_tasks,
            "passed": results.passed,
            "failed": results.failed,
            "pass_rate": results.pass_rate,
            "weighted_score": results.weighted_score,
            "execution_time_sec": results.execution_time_sec,
            "results": [
                {
                    "task_id": r.task_id,
                    "success": r.success,
                    "iterations": r.iterations,
                    "passed": r.passed,
                    "failed": r.failed,
                    "total": r.total
                }
                for r in results.results
            ]
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Always exit with 0 to allow workflow to continue
    import sys
    sys.exit(0)


if __name__ == "__main__":
    main()
