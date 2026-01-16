"""
Verilog Evaluation Engine for ArchXBench Green Agent

Uses:
- Icarus Verilog (iverilog) for compilation and vvp for simulation
- Yosys for synthesis and PPA (Power, Performance, Area) analysis
- Architectural validator for design spec compliance (rule-based + LLM)
- Parses testbench output for pass/fail counts in JSON format
"""

import os
import re
import subprocess
import tempfile
import time
import shutil
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from .models import EvaluationResult, PPAMetrics
from .arch_validator import get_validator
from .llm_validator import get_llm_validator, merge_validation_results
from .feedback_generator import get_feedback_generator
from .llm_testbench_parser import get_llm_parser


class VerilogEvaluator:
    """Evaluates Verilog submissions using Icarus Verilog and Yosys"""
    
    # Compilation timeout in seconds
    COMPILE_TIMEOUT = 30
    # Simulation timeout in seconds  
    SIMULATE_TIMEOUT = 120
    # Synthesis timeout in seconds
    SYNTHESIS_TIMEOUT = 60
    
    def __init__(self):
        self._verify_iverilog_installed()
        self._yosys_available = self._check_yosys_installed()
        self._feedback_generator = get_feedback_generator()
        self._llm_parser = get_llm_parser()
    
    def _verify_iverilog_installed(self):
        """Verify that iverilog and vvp are available"""
        try:
            result = subprocess.run(
                ["iverilog", "-V"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("iverilog not functioning properly")
        except FileNotFoundError:
            raise RuntimeError(
                "Icarus Verilog (iverilog) not found. "
                "Please install: brew install icarus-verilog (macOS) or "
                "apt install iverilog (Linux)"
            )
    
    def _check_yosys_installed(self) -> bool:
        """Check if Yosys is available for synthesis"""
        try:
            result = subprocess.run(
                ["yosys", "-V"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def evaluate(
        self, 
        task_id: str, 
        verilog_code: str, 
        testbench: str,
        run_synthesis: bool = True,
        validate_architecture: bool = True,
        use_llm_validation: bool = False,
        generate_feedback: bool = False,
        problem_description: Optional[str] = None,
        design_specs: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a Verilog submission against a testbench.
        
        Args:
            task_id: Unique identifier for the task
            verilog_code: The submitted Verilog design code
            testbench: The testbench Verilog code
            run_synthesis: Whether to run synthesis for PPA metrics
            validate_architecture: Whether to validate architectural compliance
            use_llm_validation: Whether to use LLM for advanced validation
            generate_feedback: Whether to generate detailed feedback for the agent
            problem_description: Problem description text (for constraint validation)
            design_specs: Design specification text (for interface validation)
            
        Returns:
            EvaluationResult with pass/fail counts and status
        """
        start_time = time.time()
        
        with tempfile.TemporaryDirectory(prefix="archxbench_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Write design under test
            dut_path = tmpdir_path / "design.v"
            dut_path.write_text(verilog_code)
            
            # Write testbench
            tb_path = tmpdir_path / "tb.v"
            tb_path.write_text(testbench)
            
            # Compilation phase
            compile_result = self._compile(dut_path, tb_path, tmpdir_path)
            if not compile_result[0]:
                execution_time = (time.time() - start_time) * 1000
                result = EvaluationResult(
                    task_id=task_id,
                    passed=0,
                    failed=0,
                    total=0,
                    success=False,
                    compilation_success=False,
                    simulation_success=False,
                    error_message=compile_result[1],
                    stderr_output=compile_result[2],
                    execution_time_ms=execution_time
                )
                
                # Generate feedback for compilation failures
                if generate_feedback:
                    result.feedback = self._feedback_generator.generate_feedback(
                        compilation_success=False,
                        simulation_success=False,
                        stderr_output=compile_result[2],
                        simulation_log=None,
                        passed=0,
                        failed=0,
                        total=0
                    )
                
                return result
            
            # Simulation phase
            sim_out_path = tmpdir_path / "sim.out"
            sim_result = self._simulate(sim_out_path)
            
            execution_time = (time.time() - start_time) * 1000
            
            if not sim_result[0]:
                result = EvaluationResult(
                    task_id=task_id,
                    passed=0,
                    failed=0,
                    total=0,
                    success=False,
                    compilation_success=True,
                    simulation_success=False,
                    error_message=sim_result[1],
                    simulation_log=sim_result[2],
                    execution_time_ms=execution_time
                )
                
                # Generate feedback for simulation failures
                if generate_feedback:
                    result.feedback = self._feedback_generator.generate_feedback(
                        compilation_success=True,
                        simulation_success=False,
                        stderr_output=None,
                        simulation_log=sim_result[2],
                        passed=0,
                        failed=0,
                        total=0
                    )
                
                return result
            
            # For level-4, level-5, and level-6 tasks, run Python comparison script
            sim_output = sim_result[2]
            if any(task_id.startswith(f"level-{i}/") for i in [4, 5, 6]):
                comparison_result = self._run_level6_comparison(task_id, tmpdir_path)
                if comparison_result:
                    # Append comparison output to simulation log
                    sim_output = sim_output + "\n" + comparison_result
            
            # Parse functional results
            result = self._parse_results(
                task_id, 
                sim_output, 
                execution_time
            )
            
            # Run architectural validation (if enabled and specs provided)
            if validate_architecture and problem_description and design_specs:
                # Rule-based validation (fast, objective)
                validator = get_validator()
                rule_based_compliance = validator.validate(
                    verilog_code,
                    problem_description,
                    design_specs
                )
                
                # LLM-based validation (smart, subjective) - optional
                llm_compliance = None
                if use_llm_validation:
                    llm_validator = get_llm_validator()
                    if llm_validator and llm_validator.is_available():
                        llm_compliance = llm_validator.validate(
                            verilog_code,
                            problem_description,
                            design_specs
                        )
                
                # Merge results
                result.architectural_compliance = merge_validation_results(
                    rule_based_compliance,
                    llm_compliance
                )
            
            # Run synthesis for PPA analysis (only if functional tests pass and Yosys available)
            if run_synthesis and self._yosys_available:
                ppa_metrics = self._synthesize(dut_path, tmpdir_path)
                result.ppa_metrics = ppa_metrics
            
            # Generate detailed feedback if requested
            if generate_feedback:
                result.feedback = self._feedback_generator.generate_feedback(
                    compilation_success=result.compilation_success,
                    simulation_success=result.simulation_success,
                    stderr_output=result.stderr_output,
                    simulation_log=result.simulation_log,
                    passed=result.passed,
                    failed=result.failed,
                    total=result.total
                )
            
            return result
    
    def _compile(
        self, 
        dut_path: Path, 
        tb_path: Path, 
        tmpdir: Path
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Compile Verilog files with iverilog.
        
        Returns:
            (success, error_message, stderr)
        """
        out_path = tmpdir / "sim.out"
        
        compile_cmd = [
            "iverilog",
            "-g2012",           # SystemVerilog 2012 support
            "-Wall",            # All warnings
            "-o", str(out_path),
            str(dut_path),
            str(tb_path)
        ]
        
        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=self.COMPILE_TIMEOUT
            )
            
            if result.returncode != 0:
                error_msg = "Compilation failed"
                if result.stderr:
                    # Extract key error lines
                    error_lines = [
                        line for line in result.stderr.split('\n')
                        if 'error' in line.lower() or 'Error' in line
                    ]
                    if error_lines:
                        error_msg = f"Compilation failed: {error_lines[0]}"
                
                return (False, error_msg, result.stderr)
            
            return (True, None, result.stderr)
            
        except subprocess.TimeoutExpired:
            return (False, "Compilation timeout exceeded", None)
        except Exception as e:
            return (False, f"Compilation error: {str(e)}", None)
    
    def _simulate(
        self, 
        sim_out_path: Path
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Run simulation with vvp.
        
        Returns:
            (success, error_message, output)
        """
        try:
            result = subprocess.run(
                ["vvp", str(sim_out_path)],
                capture_output=True,
                text=True,
                timeout=self.SIMULATE_TIMEOUT
            )
            
            output = result.stdout + result.stderr
            
            # Check for fatal errors (testbench assertion failures)
            # Note: $fatal returns non-zero but we still want to parse results
            if "fatal" in output.lower() and "Tests failed" in output:
                # This is expected for failing tests - still parse results
                return (True, None, output)
            
            if result.returncode != 0 and "fatal" not in output.lower():
                return (False, "Simulation failed", output)
            
            return (True, None, output)
            
        except subprocess.TimeoutExpired:
            return (False, "Simulation timeout exceeded", None)
        except Exception as e:
            return (False, f"Simulation error: {str(e)}", None)
    
    def _synthesize(
        self,
        dut_path: Path,
        tmpdir: Path
    ) -> PPAMetrics:
        """
        Synthesize Verilog design with Yosys and extract PPA metrics.
        
        Uses Yosys to:
        1. Read and elaborate the design
        2. Synthesize to generic gate library
        3. Extract gate count, cell breakdown, and estimated area
        
        Returns:
            PPAMetrics with synthesis results
        """
        metrics = PPAMetrics()
        
        # Create Yosys script
        yosys_script = tmpdir / "synth.ys"
        json_output = tmpdir / "stats.json"
        
        # Yosys synthesis script for generic technology
        script_content = f"""
# Read Verilog
read_verilog -sv {dut_path}

# Elaborate design hierarchy
hierarchy -check -auto-top

# Convert processes to netlist elements
proc

# Optimize
opt
opt_clean

# Flatten hierarchy for accurate count
flatten

# Technology mapping to generic library
techmap

# Final optimization
opt
opt_clean

# Generate statistics in JSON format
tee -q -o {json_output} stat -json

# Also get wire count
stat
"""
        
        yosys_script.write_text(script_content)
        
        try:
            result = subprocess.run(
                ["yosys", "-q", "-s", str(yosys_script)],
                capture_output=True,
                text=True,
                timeout=self.SYNTHESIS_TIMEOUT,
                cwd=str(tmpdir)
            )
            
            if result.returncode != 0:
                metrics.synthesis_success = False
                metrics.synthesis_error = f"Synthesis failed: {result.stderr[:500]}"
                metrics.synthesis_log = result.stdout + result.stderr
                return metrics
            
            # Parse JSON statistics
            if json_output.exists():
                try:
                    stats = json.loads(json_output.read_text())
                    metrics = self._parse_yosys_stats(stats, result.stdout)
                    metrics.synthesis_success = True
                    metrics.synthesis_log = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
                except json.JSONDecodeError as e:
                    metrics.synthesis_success = False
                    metrics.synthesis_error = f"Failed to parse synthesis stats: {str(e)}"
            else:
                # Fallback: parse text output
                metrics = self._parse_yosys_text_output(result.stdout)
                metrics.synthesis_success = True
                metrics.synthesis_log = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
            
            return metrics
            
        except subprocess.TimeoutExpired:
            metrics.synthesis_success = False
            metrics.synthesis_error = "Synthesis timeout exceeded"
            return metrics
        except Exception as e:
            metrics.synthesis_success = False
            metrics.synthesis_error = f"Synthesis error: {str(e)}"
            return metrics
    
    def _parse_yosys_stats(self, stats: Dict[str, Any], text_output: str) -> PPAMetrics:
        """Parse Yosys JSON statistics output"""
        metrics = PPAMetrics()
        
        # Navigate the Yosys JSON structure
        # Format: {"design": {"num_cells": N, "num_wires": N, ...}, "modules": {...}}
        design_stats = stats.get("design", {})
        
        metrics.cell_count = design_stats.get("num_cells", 0)
        metrics.wire_count = design_stats.get("num_wires", 0)
        
        # Get cell breakdown from modules section
        modules = stats.get("modules", {})
        cell_breakdown = {}
        
        for module_name, module_data in modules.items():
            cells = module_data.get("num_cells_by_type", {})
            for cell_type, count in cells.items():
                cell_breakdown[cell_type] = cell_breakdown.get(cell_type, 0) + count
        
        metrics.cell_breakdown = cell_breakdown
        
        # Calculate gate count (sum of all cell instances)
        metrics.gate_count = sum(cell_breakdown.values())
        
        # Estimate area (very rough estimate: ~1 um² per gate in generic lib)
        metrics.area_um2 = metrics.gate_count * 1.0
        
        # Parse wire count from text output if not in JSON
        if metrics.wire_count == 0:
            wire_match = re.search(r'Number of wires:\s*(\d+)', text_output)
            if wire_match:
                metrics.wire_count = int(wire_match.group(1))
        
        return metrics
    
    def _parse_yosys_text_output(self, output: str) -> PPAMetrics:
        """Fallback parser for Yosys text output"""
        metrics = PPAMetrics()
        
        # Parse cell count
        cell_match = re.search(r'Number of cells:\s*(\d+)', output)
        if cell_match:
            metrics.cell_count = int(cell_match.group(1))
            metrics.gate_count = metrics.cell_count
        
        # Parse wire count
        wire_match = re.search(r'Number of wires:\s*(\d+)', output)
        if wire_match:
            metrics.wire_count = int(wire_match.group(1))
        
        # Parse individual cell types
        cell_breakdown = {}
        cell_type_pattern = r'\$(\w+)\s+(\d+)'
        for match in re.finditer(cell_type_pattern, output):
            cell_type = f"${match.group(1)}"
            count = int(match.group(2))
            cell_breakdown[cell_type] = count
        
        metrics.cell_breakdown = cell_breakdown
        
        # Rough area estimate
        metrics.area_um2 = metrics.gate_count * 1.0
        
        return metrics

    def _run_level6_comparison(self, task_id: str, tmpdir: Path) -> Optional[str]:
        """
        Run Python comparison script for level-6 tasks.
        
        Level-6 tasks generate JSON output files during simulation,
        then use Python scripts to compare against golden outputs.
        
        Args:
            task_id: Task identifier (e.g., "level-6/aes_encryption")
            tmpdir: Temporary directory where simulation ran
            
        Returns:
            Output from the comparison script, or None if not applicable
        """
        try:
            # Get the benchmark root and task path
            # Assume benchmark tasks are in current working directory structure
            task_parts = task_id.split("/")
            if len(task_parts) != 2:
                return None
            
            level, problem_name = task_parts
            
            # Find the task directory in the workspace
            # Try common locations
            possible_roots = [
                Path.cwd(),
                Path(__file__).parent.parent,
            ]
            
            task_path = None
            for root in possible_roots:
                candidate = root / level / problem_name
                if candidate.exists():
                    task_path = candidate
                    break
            
            if not task_path:
                return None
            
            # Check if comparison script exists
            compare_script = task_path / "scripts" / "compare_outputs.py"
            if not compare_script.exists():
                return None
            
            # Copy necessary files from task directory to tmpdir
            # Level-6 tasks need: inputs/, outputs/golden_output.json, scripts/
            inputs_dir = task_path / "inputs"
            outputs_dir = task_path / "outputs"
            
            if inputs_dir.exists():
                shutil.copytree(inputs_dir, tmpdir / "inputs", dirs_exist_ok=True)
            
            if outputs_dir.exists():
                # Only copy golden outputs, not dut outputs (those are generated)
                (tmpdir / "outputs").mkdir(exist_ok=True)
                for golden_file in outputs_dir.glob("golden_*.json"):
                    shutil.copy(golden_file, tmpdir / "outputs" / golden_file.name)
            
            # Copy comparison script
            (tmpdir / "scripts").mkdir(exist_ok=True)
            shutil.copy(compare_script, tmpdir / "scripts" / "compare_outputs.py")
            
            # Run the comparison script from tmpdir
            result = subprocess.run(
                ["python3", "scripts/compare_outputs.py"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Comparison script failed:\n{result.stderr}"
                
        except Exception as e:
            return f"Error running level-6 comparison: {str(e)}"

    def _parse_results(
        self, 
        task_id: str, 
        output: str,
        execution_time: float
    ) -> EvaluationResult:
        """
        Parse testbench output for pass/fail counts.
        
        Tries multiple strategies:
        1. JSON format from testbench
        2. "Passed: X, Failed: Y" format  
        3. Rule-based patterns
        4. LLM-based parsing (if available)
        """
        # Primary: JSON format from testbench
        json_pattern = r'\{"module":\s*"[^"]*",\s*"passed":\s*(\d+),\s*"failed":\s*(\d+)\}'
        match = re.search(json_pattern, output)
        
        if match:
            passed = int(match.group(1))
            failed = int(match.group(2))
            return EvaluationResult(
                task_id=task_id,
                passed=passed,
                failed=failed,
                total=passed + failed,
                success=(failed == 0),
                compilation_success=True,
                simulation_success=True,
                simulation_log=output[-3000:] if len(output) > 3000 else output,
                execution_time_ms=execution_time
            )
        
        # Fallback: Look for "Passed: X, Failed: Y" format
        summary_pattern = r'Passed:\s*(\d+),\s*Failed:\s*(\d+)'
        match = re.search(summary_pattern, output)
        
        if match:
            passed = int(match.group(1))
            failed = int(match.group(2))
            return EvaluationResult(
                task_id=task_id,
                passed=passed,
                failed=failed,
                total=passed + failed,
                success=(failed == 0),
                compilation_success=True,
                simulation_success=True,
                simulation_log=output[-3000:] if len(output) > 3000 else output,
                execution_time_ms=execution_time
            )
        
        # Fallback 2: Look for "PASS = X, FAILED = Y" format (common in testbenches)
        summary_pattern2 = r'PASS\s*=\s*(\d+),\s*FAILED\s*=\s*(\d+)'
        match = re.search(summary_pattern2, output, re.IGNORECASE)
        
        if match:
            passed = int(match.group(1))
            failed = int(match.group(2))
            return EvaluationResult(
                task_id=task_id,
                passed=passed,
                failed=failed,
                total=passed + failed,
                success=(failed == 0),
                compilation_success=True,
                simulation_success=True,
                simulation_log=output[-3000:] if len(output) > 3000 else output,
                execution_time_ms=execution_time
            )
        
        # Fallback 3: Use LLM parser (rule-based + LLM if needed)
        if self._llm_parser:
            try:
                passed, failed, total = self._llm_parser.parse_testbench_output(output, task_id)
                return EvaluationResult(
                    task_id=task_id,
                    passed=passed,
                    failed=failed,
                    total=total,
                    success=(failed == 0 and total > 0),
                    compilation_success=True,
                    simulation_success=True,
                    simulation_log=output[-3000:] if len(output) > 3000 else output,
                    execution_time_ms=execution_time
                )
            except Exception as e:
                print(f"Warning: LLM parser failed for {task_id}: {e}")
        
        # Check for obvious failures
        if "$fatal" in output or "[FAIL]" in output:
            # Count FAIL occurrences
            fail_count = output.count("[FAIL]")
            pass_count = output.count("[PASS]") if "[PASS]" in output else 0
            
            return EvaluationResult(
                task_id=task_id,
                passed=pass_count,
                failed=max(fail_count, 1),
                total=pass_count + max(fail_count, 1),
                success=False,
                compilation_success=True,
                simulation_success=True,
                error_message="Test assertions failed",
                simulation_log=output[-3000:] if len(output) > 3000 else output,
                execution_time_ms=execution_time
            )
        
        # Could not parse - return unknown state
        return EvaluationResult(
            task_id=task_id,
            passed=0,
            failed=0,
            total=0,
            success=False,
            compilation_success=True,
            simulation_success=True,
            error_message="Could not parse test results from simulation output",
            simulation_log=output[-3000:] if len(output) > 3000 else output,
            execution_time_ms=execution_time
        )


# Singleton evaluator instance
_evaluator: Optional[VerilogEvaluator] = None


def get_evaluator() -> VerilogEvaluator:
    """Get or create the singleton evaluator instance"""
    global _evaluator
    if _evaluator is None:
        _evaluator = VerilogEvaluator()
    return _evaluator
