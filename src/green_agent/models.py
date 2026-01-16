"""
Data models for ArchXBench Green Agent
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class TaskLevel(str, Enum):
    """Benchmark complexity levels"""
    LEVEL_0 = "level-0"   # Logic Building Blocks
    LEVEL_1A = "level-1a" # Simple Arithmetic
    LEVEL_1B = "level-1b" # Hierarchical/Parametric
    LEVEL_1C = "level-1c" # Complex Arithmetic
    LEVEL_2 = "level-2"   # Pipelined Integer
    LEVEL_3 = "level-3"   # Iterative FP/Fixed-Point
    LEVEL_4 = "level-4"   # Pipelined FP/DSP
    LEVEL_5 = "level-5"   # Streaming/Systolic
    LEVEL_6 = "level-6"   # Domain-Specific Accelerators


@dataclass
class ArchitecturalCompliance:
    """Tracks compliance with design specs and architectural constraints"""
    # Overall compliance
    is_compliant: bool = True
    compliance_score: float = 1.0  # 0.0 to 1.0
    
    # Constraint violations
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Specific checks
    structural_compliance: bool = True  # Gate-level vs behavioral
    interface_compliance: bool = True   # Correct inputs/outputs
    hierarchy_compliance: bool = True   # Module structure
    constraint_compliance: bool = True  # Design constraints met
    
    # Details
    required_style: Optional[str] = None  # e.g., "gate-level", "behavioral"
    detected_style: Optional[str] = None
    required_constraints: List[str] = field(default_factory=list)
    validation_log: Optional[str] = None
    
    def add_violation(self, message: str):
        """Add a compliance violation"""
        self.violations.append(message)
        self.is_compliant = False
        self._update_score()
    
    def add_warning(self, message: str):
        """Add a compliance warning"""
        self.warnings.append(message)
        self._update_score()
    
    def _update_score(self):
        """Update compliance score based on violations and warnings"""
        if not self.is_compliant:
            # Each violation reduces score
            penalty = min(len(self.violations) * 0.25, 1.0)
            self.compliance_score = max(0.0, 1.0 - penalty)
        else:
            # Warnings only slightly reduce score
            penalty = min(len(self.warnings) * 0.1, 0.3)
            self.compliance_score = max(0.7, 1.0 - penalty)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_compliant": self.is_compliant,
            "compliance_score": self.compliance_score,
            "violations": self.violations,
            "warnings": self.warnings,
            "structural_compliance": self.structural_compliance,
            "interface_compliance": self.interface_compliance,
            "hierarchy_compliance": self.hierarchy_compliance,
            "constraint_compliance": self.constraint_compliance,
            "required_style": self.required_style,
            "detected_style": self.detected_style,
            "required_constraints": self.required_constraints
        }


@dataclass
class BenchmarkTask:
    """Represents a single benchmark task"""
    task_id: str
    level: str
    problem_name: str
    problem_description: str
    design_specs: str
    testbench: str
    difficulty_score: int = 1  # 1-10 scale based on level
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "level": self.level,
            "problem_name": self.problem_name,
            "problem_description": self.problem_description,
            "design_specs": self.design_specs,
            "testbench": self.testbench,
            "difficulty_score": self.difficulty_score
        }


@dataclass
class PPAMetrics:
    """Power, Performance, Area metrics from synthesis"""
    # Area metrics
    gate_count: int = 0
    cell_count: int = 0
    wire_count: int = 0
    area_um2: float = 0.0
    
    # Timing/Performance metrics
    critical_path_ns: float = 0.0
    max_frequency_mhz: float = 0.0
    
    # Power estimates (simplified - actual power needs more info)
    estimated_power_uw: float = 0.0
    
    # Detailed breakdown
    cell_breakdown: Dict[str, int] = field(default_factory=dict)
    
    # Synthesis metadata
    technology: str = "generic"  # e.g., "sky130", "generic"
    synthesis_tool: str = "yosys"
    synthesis_success: bool = False
    synthesis_error: Optional[str] = None
    synthesis_log: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_count": self.gate_count,
            "cell_count": self.cell_count,
            "wire_count": self.wire_count,
            "area_um2": self.area_um2,
            "critical_path_ns": self.critical_path_ns,
            "max_frequency_mhz": self.max_frequency_mhz,
            "estimated_power_uw": self.estimated_power_uw,
            "cell_breakdown": self.cell_breakdown,
            "technology": self.technology,
            "synthesis_tool": self.synthesis_tool,
            "synthesis_success": self.synthesis_success,
            "synthesis_error": self.synthesis_error
        }


@dataclass
class TestFailure:
    """Details about a single test failure"""
    test_number: int
    test_description: str
    expected_output: Optional[str] = None
    actual_output: Optional[str] = None
    error_type: str = "functional"  # functional, timing, assertion
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_number": self.test_number,
            "test_description": self.test_description,
            "expected_output": self.expected_output,
            "actual_output": self.actual_output,
            "error_type": self.error_type
        }


@dataclass
class FeedbackDetails:
    """Detailed feedback for purple agent to improve design"""
    # Compilation feedback
    compilation_errors: List[str] = field(default_factory=list)
    compilation_warnings: List[str] = field(default_factory=list)
    
    # Simulation feedback
    test_failures: List[TestFailure] = field(default_factory=list)
    failed_test_numbers: List[int] = field(default_factory=list)
    
    # Behavioral feedback
    behavioral_issues: List[str] = field(default_factory=list)
    
    # Suggestions for improvement
    suggestions: List[str] = field(default_factory=list)
    
    # Raw logs (for advanced agents)
    compilation_log: Optional[str] = None
    simulation_log: Optional[str] = None
    
    def has_feedback(self) -> bool:
        """Check if there is any actionable feedback"""
        return bool(
            self.compilation_errors or 
            self.test_failures or 
            self.behavioral_issues
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "compilation_errors": self.compilation_errors,
            "compilation_warnings": self.compilation_warnings,
            "test_failures": [tf.to_dict() for tf in self.test_failures],
            "failed_test_numbers": self.failed_test_numbers,
            "behavioral_issues": self.behavioral_issues,
            "suggestions": self.suggestions,
            "compilation_log": self.compilation_log,
            "simulation_log": self.simulation_log
        }


@dataclass
class EvaluationResult:
    """Result of evaluating a purple agent's submission"""
    task_id: str
    passed: int
    failed: int
    total: int
    success: bool
    compilation_success: bool = True
    simulation_success: bool = True
    error_message: Optional[str] = None
    stderr_output: Optional[str] = None
    simulation_log: Optional[str] = None
    execution_time_ms: float = 0.0
    
    # PPA metrics (optional, populated when synthesis is run)
    ppa_metrics: Optional[PPAMetrics] = None
    
    # Architectural compliance (optional, populated when validation is run)
    architectural_compliance: Optional[ArchitecturalCompliance] = None
    
    # Detailed feedback (optional, populated when feedback is requested)
    feedback: Optional[FeedbackDetails] = None
    
    @property
    def synthesis_success(self) -> bool:
        """Check if synthesis was successful"""
        return self.ppa_metrics is not None and self.ppa_metrics.synthesis_success
    
    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "task_id": self.task_id,
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
            "success": self.success,
            "pass_rate": self.pass_rate,
            "compilation_success": self.compilation_success,
            "simulation_success": self.simulation_success,
            "synthesis_success": self.synthesis_success,
            "error_message": self.error_message,
            "stderr_output": self.stderr_output,
            "simulation_log": self.simulation_log,
            "execution_time_ms": self.execution_time_ms,
            "ppa_metrics": self.ppa_metrics.to_dict() if self.ppa_metrics else None,
            "architectural_compliance": self.architectural_compliance.to_dict() if self.architectural_compliance else None,
            "feedback": self.feedback.to_dict() if self.feedback else None
        }
        return result


@dataclass
class TaskSubmission:
    """Submission from a purple agent"""
    task_id: str
    verilog_code: str
    agent_id: Optional[str] = None
    submission_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkMetrics:
    """Overall benchmark metrics and metadata"""
    benchmark_name: str = "ArchXBench"
    version: str = "1.0.0"
    total_tasks: int = 0
    levels: Dict[str, str] = field(default_factory=dict)
    tasks_per_level: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "version": self.version,
            "description": "RTL Synthesis Benchmark for LLM-driven Verilog generation",
            "total_tasks": self.total_tasks,
            "levels": self.levels,
            "tasks_per_level": self.tasks_per_level,
            "scoring": {
                "metric": "pass_rate",
                "formula": "passed_tests / total_tests per task",
                "aggregation": "weighted mean across all tasks by difficulty",
                "difficulty_weights": {
                    "level-0": 1, "level-1a": 2, "level-1b": 3,
                    "level-1c": 4, "level-2": 5, "level-3": 6,
                    "level-4": 7, "level-5": 8, "level-6": 10
                }
            },
            "evaluation": {
                "tool": "Icarus Verilog (iverilog)",
                "version": ">=11.0",
                "timeout_compile_sec": 30,
                "timeout_simulate_sec": 120
            }
        }


@dataclass 
class AgentSession:
    """Tracks a purple agent's evaluation session"""
    session_id: str
    agent_id: str
    started_at: str
    tasks_attempted: List[str] = field(default_factory=list)
    tasks_passed: List[str] = field(default_factory=list)
    results: List[EvaluationResult] = field(default_factory=list)
    
    @property
    def overall_pass_rate(self) -> float:
        if not self.results:
            return 0.0
        total_passed = sum(r.passed for r in self.results)
        total_tests = sum(r.total for r in self.results)
        return total_passed / total_tests if total_tests > 0 else 0.0
