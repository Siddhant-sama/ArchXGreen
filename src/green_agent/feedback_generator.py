"""
Feedback Generator for ArchXBench

Analyzes evaluation results and generates detailed feedback for purple agents
to help them improve their Verilog designs iteratively.
"""

import re
from typing import List, Dict, Optional
from .models import FeedbackDetails, TestFailure


class FeedbackGenerator:
    """
    Generates structured feedback from evaluation logs
    """
    
    def __init__(self):
        pass
    
    def generate_feedback(
        self,
        compilation_success: bool,
        simulation_success: bool,
        stderr_output: Optional[str],
        simulation_log: Optional[str],
        passed: int,
        failed: int,
        total: int
    ) -> FeedbackDetails:
        """
        Generate detailed feedback from evaluation results
        
        Args:
            compilation_success: Whether compilation succeeded
            simulation_success: Whether simulation succeeded
            stderr_output: stderr from compilation
            simulation_log: stdout from simulation
            passed: Number of tests passed
            failed: Number of tests failed
            total: Total number of tests
            
        Returns:
            FeedbackDetails with structured feedback
        """
        feedback = FeedbackDetails()
        
        # 1. Parse compilation errors and warnings
        if not compilation_success and stderr_output:
            feedback.compilation_errors = self._parse_compilation_errors(stderr_output)
            feedback.compilation_warnings = self._parse_compilation_warnings(stderr_output)
            feedback.compilation_log = stderr_output
            
            # Add suggestions for common compilation issues
            feedback.suggestions.extend(self._suggest_compilation_fixes(feedback.compilation_errors))
        
        # 2. Parse simulation failures
        if simulation_success and failed > 0 and simulation_log:
            feedback.test_failures = self._parse_test_failures(simulation_log)
            feedback.failed_test_numbers = [tf.test_number for tf in feedback.test_failures]
            feedback.simulation_log = simulation_log
            
            # Add suggestions for test failures
            feedback.suggestions.extend(self._suggest_test_fixes(feedback.test_failures))
        
        # 3. Parse behavioral issues
        if simulation_log:
            feedback.behavioral_issues = self._parse_behavioral_issues(simulation_log)
        
        # 4. Add general suggestions based on pass rate
        if total > 0:
            pass_rate = passed / total
            if pass_rate == 0:
                feedback.suggestions.append(
                    "All tests failed. Review the problem description and design specs carefully. "
                    "Start with simple test cases first."
                )
            elif 0 < pass_rate < 0.5:
                feedback.suggestions.append(
                    "Less than half of tests passed. Check for logic errors in your implementation."
                )
            elif 0.5 <= pass_rate < 1.0:
                feedback.suggestions.append(
                    "Some tests passing. Focus on the failing test cases to identify edge cases or timing issues."
                )
        
        return feedback
    
    def _parse_compilation_errors(self, stderr: str) -> List[str]:
        """Extract compilation errors from stderr"""
        errors = []
        
        # iverilog error patterns
        # Example: design.v:10: error: syntax error
        error_pattern = r'(?:design\.v|tb\.v):(\d+):\s*error:\s*(.+)'
        
        for match in re.finditer(error_pattern, stderr, re.MULTILINE):
            line_num = match.group(1)
            error_msg = match.group(2).strip()
            errors.append(f"Line {line_num}: {error_msg}")
        
        # If no structured errors found, include raw stderr (first 500 chars)
        if not errors and stderr:
            truncated = stderr[:500] + ("..." if len(stderr) > 500 else "")
            errors.append(f"Compilation failed with: {truncated}")
        
        return errors
    
    def _parse_compilation_warnings(self, stderr: str) -> List[str]:
        """Extract compilation warnings from stderr"""
        warnings = []
        
        # iverilog warning patterns
        warning_pattern = r'(?:design\.v|tb\.v):(\d+):\s*warning:\s*(.+)'
        
        for match in re.finditer(warning_pattern, stderr, re.MULTILINE):
            line_num = match.group(1)
            warning_msg = match.group(2).strip()
            warnings.append(f"Line {line_num}: {warning_msg}")
        
        return warnings
    
    def _parse_test_failures(self, simulation_log: str) -> List[TestFailure]:
        """
        Extract detailed test failure information from simulation log
        
        Looks for patterns like:
        - Test #X: FAIL
        - ERROR at time Y: Expected Z, got W
        - Assertion failed: ...
        """
        failures = []
        
        # Pattern 1: Test #X: FAIL or Test X FAIL
        test_fail_pattern = r'Test\s*#?(\d+):\s*FAIL'
        
        for match in re.finditer(test_fail_pattern, simulation_log, re.IGNORECASE):
            test_num = int(match.group(1))
            
            # Try to extract context around this failure
            context = self._extract_context(simulation_log, match.start(), match.end())
            
            # Look for expected vs actual values in context
            expected, actual = self._extract_expected_actual(context)
            
            failure = TestFailure(
                test_number=test_num,
                test_description=f"Test {test_num} failed",
                expected_output=expected,
                actual_output=actual,
                error_type="functional"
            )
            failures.append(failure)
        
        # Pattern 2: ERROR messages
        error_pattern = r'ERROR[:\s]+(.+?)(?:\n|$)'
        
        for i, match in enumerate(re.finditer(error_pattern, simulation_log, re.IGNORECASE)):
            error_msg = match.group(1).strip()
            
            # Try to extract test number from nearby context
            test_num = self._extract_test_number_from_context(
                simulation_log,
                match.start() - 100,
                match.start()
            )
            
            if test_num is None:
                test_num = i + 1  # Generic test number
            
            # Check if we already have this test failure
            existing = next((f for f in failures if f.test_number == test_num), None)
            if existing:
                # Enhance existing failure
                if not existing.test_description or "failed" in existing.test_description.lower():
                    existing.test_description = error_msg
            else:
                failure = TestFailure(
                    test_number=test_num,
                    test_description=error_msg,
                    error_type="functional"
                )
                failures.append(failure)
        
        # Pattern 3: Assertion failures
        assertion_pattern = r'Assertion\s+failed[:\s]+(.+?)(?:\n|$)'
        
        for match in re.finditer(assertion_pattern, simulation_log, re.IGNORECASE):
            assertion_msg = match.group(1).strip()
            
            test_num = self._extract_test_number_from_context(
                simulation_log,
                match.start() - 100,
                match.start()
            )
            
            if test_num is None:
                test_num = len(failures) + 1
            
            failure = TestFailure(
                test_number=test_num,
                test_description=f"Assertion failed: {assertion_msg}",
                error_type="assertion"
            )
            failures.append(failure)
        
        return failures
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 200) -> str:
        """Extract context around a match"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _extract_expected_actual(self, context: str) -> tuple[Optional[str], Optional[str]]:
        """Extract expected and actual values from context"""
        expected = None
        actual = None
        
        # Pattern: Expected: X, Got: Y
        pattern1 = r'Expected[:\s]+([^\s,]+).*?(?:Got|Actual)[:\s]+([^\s,]+)'
        match = re.search(pattern1, context, re.IGNORECASE)
        if match:
            expected = match.group(1)
            actual = match.group(2)
            return expected, actual
        
        # Pattern: Expected X but got Y
        pattern2 = r'Expected\s+(\S+)\s+but\s+got\s+(\S+)'
        match = re.search(pattern2, context, re.IGNORECASE)
        if match:
            expected = match.group(1)
            actual = match.group(2)
            return expected, actual
        
        return expected, actual
    
    def _extract_test_number_from_context(
        self,
        text: str,
        start: int,
        end: int
    ) -> Optional[int]:
        """Try to find a test number in nearby context"""
        if start < 0:
            start = 0
        context = text[start:end]
        
        match = re.search(r'Test\s*#?(\d+)', context, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        return None
    
    def _parse_behavioral_issues(self, simulation_log: str) -> List[str]:
        """Extract behavioral issues from simulation log"""
        issues = []
        
        # Look for timing violations
        if "timing violation" in simulation_log.lower():
            issues.append("Timing violations detected in simulation")
        
        # Look for metastability warnings
        if "metastable" in simulation_log.lower():
            issues.append("Metastability warning detected")
        
        # Look for X (unknown) values
        if re.search(r"output.*?x+", simulation_log, re.IGNORECASE):
            issues.append("Unknown (X) values detected in output signals")
        
        # Look for Z (high-impedance) issues
        if re.search(r"output.*?z+", simulation_log, re.IGNORECASE):
            issues.append("High-impedance (Z) values in output signals (possible floating outputs)")
        
        return issues
    
    def _suggest_compilation_fixes(self, errors: List[str]) -> List[str]:
        """Generate suggestions based on compilation errors"""
        suggestions = []
        
        error_text = " ".join(errors).lower()
        
        if "syntax error" in error_text:
            suggestions.append(
                "Syntax error: Check for missing semicolons, mismatched parentheses, "
                "or incorrect Verilog keywords."
            )
        
        if "undeclared" in error_text or "not defined" in error_text:
            suggestions.append(
                "Undeclared identifier: Ensure all signals are declared with proper wire/reg types "
                "before use."
            )
        
        if "port" in error_text and "mismatch" in error_text:
            suggestions.append(
                "Port mismatch: Verify module instantiation matches the module definition. "
                "Check port names, order, and widths."
            )
        
        if "width" in error_text or "size" in error_text:
            suggestions.append(
                "Width mismatch: Check that signal widths match between assignments and "
                "module ports. Use proper bit slicing [N:0]."
            )
        
        return suggestions
    
    def _suggest_test_fixes(self, failures: List[TestFailure]) -> List[str]:
        """Generate suggestions based on test failures"""
        suggestions = []
        
        if not failures:
            return suggestions
        
        # Analyze failure patterns
        failure_descriptions = " ".join([f.test_description.lower() for f in failures])
        
        if "overflow" in failure_descriptions:
            suggestions.append(
                "Overflow detected: Check that your arithmetic operations handle carry/borrow "
                "correctly and output widths are sufficient."
            )
        
        if "timing" in failure_descriptions or "clock" in failure_descriptions:
            suggestions.append(
                "Timing issue: Verify that your sequential logic updates on the correct clock edge "
                "and all registers are properly initialized."
            )
        
        if "edge case" in failure_descriptions or "boundary" in failure_descriptions:
            suggestions.append(
                "Edge case failure: Pay special attention to boundary conditions like "
                "all-zeros, all-ones, maximum values, and minimum values."
            )
        
        # If multiple specific tests failed
        if len(failures) > 1:
            test_nums = sorted([f.test_number for f in failures])
            suggestions.append(
                f"Tests {', '.join(map(str, test_nums))} failed. "
                "Review the testbench to understand what these specific tests are checking."
            )
        
        return suggestions


def get_feedback_generator() -> FeedbackGenerator:
    """Factory function to get feedback generator instance"""
    return FeedbackGenerator()
