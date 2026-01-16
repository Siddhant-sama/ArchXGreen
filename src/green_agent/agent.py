"""
ArchXBench Green Agent - Core Benchmark Agent

This agent loads all benchmark tasks and provides the A2A interface
for purple agents to discover tasks and submit solutions.
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from .models import (
    BenchmarkTask, 
    EvaluationResult, 
    TaskSubmission,
    BenchmarkMetrics,
    AgentSession,
    TaskLevel
)
from .evaluator import get_evaluator
from .benchmark_loader import get_benchmark_loader


# Difficulty scores by level (used for weighted scoring)
LEVEL_DIFFICULTY = {
    "level-0": 1,
    "level-1a": 2,
    "level-1b": 3,
    "level-1c": 4,
    "level-2": 5,
    "level-3": 6,
    "level-4": 7,
    "level-5": 8,
    "level-6": 10
}

# Level descriptions
LEVEL_DESCRIPTIONS = {
    "level-0": "Logic Building Blocks - Fundamental combinational and sequential primitives",
    "level-1a": "Simple Arithmetic - Basic integer and bit-wise operators",
    "level-1b": "Hierarchical/Parametric - Parameterizable modules assembled hierarchically",
    "level-1c": "Complex Arithmetic - High-performance integer units (parallel-prefix, tree multipliers)",
    "level-2": "Pipelined Integer - Pipelined implementations of arithmetic units",
    "level-3": "Iterative FP/Fixed-Point - Floating-point and fixed-point iterative algorithms",
    "level-4": "Pipelined FP/DSP - Pipelined FP units and DSP blocks",
    "level-5": "Streaming/Systolic - Streaming architectures and systolic arrays",
    "level-6": "Domain-Specific Accelerators - Complex accelerators (AES, FFT, convolution)"
}


class ArchXBenchGreenAgent:
    """
    Green Agent: Benchmark Evaluator for ArchXBench RTL Synthesis Tasks
    
    This agent implements the A2A protocol to:
    1. Expose available tasks to purple agents
    2. Provide task details (problem description, specs, testbench)
    3. Evaluate submitted Verilog code against testbenches
    4. Track session results and compute metrics
    """
    
    def __init__(
        self, 
        benchmark_root: Optional[str] = None,
        use_dynamic_loader: bool = True,
        cache_dir: Optional[Path] = None,
        auto_update: bool = True
    ):
        """
        Initialize the green agent with the benchmark directory.
        
        Args:
            benchmark_root: Path to local ArchXBench directory (if not using dynamic loader)
            use_dynamic_loader: If True, fetch benchmarks from GitHub dynamically
            cache_dir: Cache directory for dynamic loader
            auto_update: Automatically update cached benchmarks if expired
        """
        if use_dynamic_loader:
            # Use dynamic loader to fetch from GitHub
            print("Initializing with dynamic benchmark loader...")
            try:
                self.loader = get_benchmark_loader(cache_dir=cache_dir, auto_update=auto_update)
                self.benchmark_root = self.loader.get_benchmark_root()
                print(f"Benchmark root: {self.benchmark_root}")
            except Exception as e:
                print(f"ERROR: Failed to load benchmarks dynamically: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            # Use static local directory
            if not benchmark_root:
                raise ValueError("benchmark_root must be provided when use_dynamic_loader=False")
            self.benchmark_root = Path(benchmark_root).resolve()
            self.loader = None
            print(f"Using static benchmark directory: {self.benchmark_root}")
        
        self.tasks: Dict[str, BenchmarkTask] = {}
        self.sessions: Dict[str, AgentSession] = {}
        
        # Load all tasks
        self._load_all_tasks()
        
        # Initialize evaluator (lazy - only when needed)
        self._evaluator = None
    
    @property
    def evaluator(self):
        """Lazy initialization of evaluator"""
        if self._evaluator is None:
            self._evaluator = get_evaluator()
        return self._evaluator
    
    def _load_all_tasks(self) -> None:
        """Load all benchmark tasks from the directory structure"""
        levels = [
            d for d in os.listdir(self.benchmark_root)
            if d.startswith("level-") and 
               os.path.isdir(os.path.join(self.benchmark_root, d))
        ]
        
        for level in sorted(levels):
            level_path = self.benchmark_root / level
            
            for problem_name in sorted(os.listdir(level_path)):
                problem_path = level_path / problem_name
                
                if not problem_path.is_dir():
                    continue
                
                task_id = f"{level}/{problem_name}"
                
                try:
                    # For level-6, testbenches may be named differently
                    # Try tb.v first (standard), then tb_*.v, then testbench.v
                    tb_file = problem_path / "tb.v"
                    if not tb_file.exists():
                        if level == "level-6" or level == "level-4" or level == "level-5":
                            # Try to find any tb_*.v file
                            tb_files = list(problem_path.glob("tb_*.v"))
                            if tb_files:
                                tb_file = tb_files[0]
                            else:
                                # Try testbench.v as fallback
                                alt_tb = problem_path / "testbench.v"
                                if alt_tb.exists():
                                    tb_file = alt_tb
                    
                    task = BenchmarkTask(
                        task_id=task_id,
                        level=level,
                        problem_name=problem_name,
                        problem_description=self._read_file(problem_path / "problem-description.txt"),
                        design_specs=self._read_file(problem_path / "design-specs.txt"),
                        testbench=self._read_file(tb_file),
                        difficulty_score=LEVEL_DIFFICULTY.get(level, 5)
                    )
                    self.tasks[task_id] = task
                except FileNotFoundError as e:
                    print(f"Warning: Skipping {task_id}: Missing required file - {e}")
                except Exception as e:
                    print(f"Warning: Error loading {task_id}: {e}")
    
    def _read_file(self, filepath: Path) -> str:
        """Read file contents"""
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    
    # ========== A2A Protocol: Task Discovery ==========
    
    def get_task_list(
        self, 
        level: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Return list of available tasks for purple agents to discover.
        
        Args:
            level: Filter by specific level (e.g., "level-0")
            limit: Maximum number of tasks to return
            offset: Pagination offset
            
        Returns:
            List of task metadata dicts
        """
        tasks = []
        
        for task_id, task in self.tasks.items():
            if level and task.level != level:
                continue
            
            tasks.append({
                "task_id": task_id,
                "level": task.level,
                "problem_name": task.problem_name,
                "difficulty_score": task.difficulty_score,
                "level_description": LEVEL_DESCRIPTIONS.get(task.level, "")
            })
        
        # Apply pagination
        if offset:
            tasks = tasks[offset:]
        if limit:
            tasks = tasks[:limit]
        
        return tasks
    
    def get_task(self, task_id: str, include_testbench: bool = True) -> Dict[str, Any]:
        """
        Return detailed task information for a purple agent to solve.
        
        Args:
            task_id: The task identifier (e.g., "level-0/mux2to1")
            include_testbench: Whether to include the testbench (for reference)
            
        Returns:
            Task details including problem description and design specs
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found")
        
        task = self.tasks[task_id]
        
        result = {
            "task_id": task_id,
            "level": task.level,
            "problem_name": task.problem_name,
            "difficulty_score": task.difficulty_score,
            "problem_description": task.problem_description,
            "design_specs": task.design_specs,
        }
        
        if include_testbench:
            result["testbench"] = task.testbench
        
        return result
    
    # ========== A2A Protocol: Evaluation ==========
    
    def evaluate_submission(
        self, 
        task_id: str, 
        verilog_code: str,
        session_id: Optional[str] = None,
        validate_architecture: bool = True,
        use_llm_validation: bool = False,
        generate_feedback: bool = False
    ) -> EvaluationResult:
        """
        Evaluate a purple agent's Verilog submission.
        
        Args:
            task_id: The task being solved
            verilog_code: The submitted Verilog code
            session_id: Optional session ID for tracking
            validate_architecture: Whether to validate architectural compliance
            use_llm_validation: Whether to use LLM for advanced validation (slower but smarter)
            generate_feedback: Whether to generate detailed feedback for improvement
            
        Returns:
            EvaluationResult with pass/fail counts and detailed feedback
        """
        if task_id not in self.tasks:
            return EvaluationResult(
                task_id=task_id,
                passed=0,
                failed=0,
                total=0,
                success=False,
                error_message=f"Task '{task_id}' not found"
            )
        
        task = self.tasks[task_id]
        
        # Run evaluation with architectural validation
        result = self.evaluator.evaluate(
            task_id=task_id,
            verilog_code=verilog_code,
            testbench=task.testbench,
            run_synthesis=True,
            validate_architecture=validate_architecture,
            use_llm_validation=use_llm_validation,
            generate_feedback=generate_feedback,
            problem_description=task.problem_description,
            design_specs=task.design_specs
        )
        
        # Track in session if provided
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            if task_id not in session.tasks_attempted:
                session.tasks_attempted.append(task_id)
            if result.success and task_id not in session.tasks_passed:
                session.tasks_passed.append(task_id)
            session.results.append(result)
        
        return result
    
    # ========== A2A Protocol: Session Management ==========
    
    def create_session(self, agent_id: str) -> str:
        """
        Create a new evaluation session for tracking agent progress.
        
        Args:
            agent_id: Identifier for the purple agent
            
        Returns:
            Session ID for subsequent API calls
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = AgentSession(
            session_id=session_id,
            agent_id=agent_id,
            started_at=datetime.utcnow().isoformat()
        )
        return session_id
    
    def get_session_results(self, session_id: str) -> Dict[str, Any]:
        """Get aggregated results for a session"""
        if session_id not in self.sessions:
            raise ValueError(f"Session '{session_id}' not found")
        
        session = self.sessions[session_id]
        
        return {
            "session_id": session_id,
            "agent_id": session.agent_id,
            "started_at": session.started_at,
            "tasks_attempted": len(session.tasks_attempted),
            "tasks_passed": len(session.tasks_passed),
            "overall_pass_rate": session.overall_pass_rate,
            "results": [r.to_dict() for r in session.results]
        }
    
    def reset_session(self, session_id: str) -> bool:
        """Reset a session to clean state (A2A reproducibility requirement)"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.tasks_attempted = []
        session.tasks_passed = []
        session.results = []
        return True
    
    # ========== A2A Protocol: Metrics ==========
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return benchmark metadata and scoring information"""
        tasks_per_level = {}
        for task in self.tasks.values():
            level = task.level
            tasks_per_level[level] = tasks_per_level.get(level, 0) + 1
        
        metrics = BenchmarkMetrics(
            total_tasks=len(self.tasks),
            levels=LEVEL_DESCRIPTIONS,
            tasks_per_level=tasks_per_level
        )
        
        return metrics.to_dict()
    
    def compute_weighted_score(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Compute weighted score across all evaluated tasks.
        
        Higher levels contribute more to the final score.
        """
        if not results:
            return {
                "weighted_score": 0.0,
                "tasks_evaluated": 0,
                "tasks_passed": 0
            }
        
        total_weight = 0
        weighted_sum = 0
        tasks_passed = 0
        
        for result in results:
            task = self.tasks.get(result.task_id)
            if not task:
                continue
            
            weight = task.difficulty_score
            total_weight += weight
            weighted_sum += result.pass_rate * weight
            
            if result.success:
                tasks_passed += 1
        
        weighted_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        return {
            "weighted_score": weighted_score,
            "tasks_evaluated": len(results),
            "tasks_passed": tasks_passed,
            "total_weight": total_weight
        }
    
    # ========== Dynamic Benchmark Management ==========
    
    def update_benchmarks(self, force: bool = False) -> Dict[str, Any]:
        """
        Update benchmarks from remote repository
        
        Args:
            force: Force update even if cache is valid
            
        Returns:
            Update status information
        """
        if not self.loader:
            return {
                "status": "error",
                "message": "Dynamic loader not enabled. Agent was initialized with static directory."
            }
        
        try:
            if force:
                self.loader.force_update()
            else:
                # Check if update is needed
                cache_info = self.loader.get_cache_info()
                if not cache_info.get('expired', False):
                    return {
                        "status": "skipped",
                        "message": "Cache is still valid, no update needed",
                        "cache_info": cache_info
                    }
                
                # Update if expired
                self.benchmark_root = self.loader.get_benchmark_root()
            
            # Reload all tasks
            self.tasks.clear()
            self._load_all_tasks()
            
            return {
                "status": "success",
                "message": f"Benchmarks updated successfully. Loaded {len(self.tasks)} tasks.",
                "tasks_loaded": len(self.tasks),
                "cache_info": self.loader.get_cache_info()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to update benchmarks: {str(e)}"
            }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached benchmarks
        
        Returns:
            Cache status and metadata
        """
        if not self.loader:
            return {
                "dynamic_loader": False,
                "message": "Using static local directory",
                "benchmark_root": str(self.benchmark_root)
            }
        
        cache_info = self.loader.get_cache_info()
        cache_info["dynamic_loader"] = True
        cache_info["tasks_loaded"] = len(self.tasks)
        return cache_info


# Factory function for easy instantiation
def create_green_agent(
    benchmark_root: Optional[str] = None,
    use_dynamic_loader: bool = True,
    cache_dir: Optional[Path] = None,
    auto_update: bool = True
) -> ArchXBenchGreenAgent:
    """Create a new ArchXBench Green Agent instance"""
    return ArchXBenchGreenAgent(
        benchmark_root=benchmark_root,
        use_dynamic_loader=use_dynamic_loader,
        cache_dir=cache_dir,
        auto_update=auto_update
    )
