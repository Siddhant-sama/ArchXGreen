"""
Interactive CLI for ArchXBench Green Agent

Allows users to solve RTL tasks manually by:
1. Browsing available tasks
2. Reading problem descriptions
3. Submitting their Verilog code
4. Getting instant evaluation feedback
5. Iterating until success

This is the standalone user interface - no purple agent required.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
import argparse

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green_agent.agent import create_green_agent


class InteractiveSession:
    """Interactive evaluation session for manual task solving"""
    
    def __init__(self, agent):
        self.agent = agent
        self.current_task = None
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name != 'nt' else 'cls')
    
    def print_header(self, title: str):
        """Print formatted section header"""
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    
    def print_task_info(self, task: dict):
        """Display task information"""
        self.print_header(f"Task: {task['task_id']}")
        print(f"📋 Problem: {task['problem_name']}")
        print(f"🎯 Level: {task['level']} (Difficulty: {task['difficulty_score']})")
        print(f"\n📖 Description:\n{'-'*70}")
        print(task['problem_description'])
        print(f"\n📐 Design Specifications:\n{'-'*70}")
        print(task['design_specs'])
        print(f"{'-'*70}\n")
    
    def print_results(self, result):
        """Display evaluation results"""
        self.print_header("Evaluation Results")
        
        # Compilation
        status = "✅" if result.compilation_success else "❌"
        print(f"{status} Compilation: {'SUCCESS' if result.compilation_success else 'FAILED'}")
        
        # Simulation
        status = "✅" if result.simulation_success else "❌"
        print(f"{status} Simulation: {'SUCCESS' if result.simulation_success else 'FAILED'}")
        
        # Synthesis (optional)
        if result.synthesis_success is not None:
            status = "✅" if result.synthesis_success else "❌"
            print(f"{status} Synthesis: {'SUCCESS' if result.synthesis_success else 'FAILED'}")
        
        # Test results
        print(f"\n📊 Test Results:")
        print(f"   Passed: {result.passed}/{result.total}")
        print(f"   Failed: {result.failed}/{result.total}")
        print(f"   Pass Rate: {result.pass_rate*100:.1f}%")
        
        # Overall success
        if result.success:
            print(f"\n🎉 {'SUCCESS! All tests passed!'}")
        else:
            print(f"\n⚠️  {'FAILED - See feedback below'}")
        
        # PPA Metrics
        if result.ppa_metrics:
            print(f"\n⚡ PPA Metrics:")
            print(f"   Gate Count: {result.ppa_metrics.gate_count}")
            if result.ppa_metrics.cell_breakdown:
                print(f"   Cell Breakdown: {result.ppa_metrics.cell_breakdown}")
        
        # Architectural Compliance
        if result.architectural_compliance:
            compliance = result.architectural_compliance
            status = "✅" if compliance.is_compliant else "❌"
            print(f"\n🏗️  Architectural Compliance: {status}")
            print(f"   Score: {compliance.score*100:.1f}%")
            if compliance.violations:
                print(f"   ⚠️  Violations:")
                for v in compliance.violations:
                    print(f"      • {v}")
            if compliance.warnings:
                print(f"   ⚠️  Warnings:")
                for w in compliance.warnings:
                    print(f"      • {w}")
        
        # Feedback
        if result.feedback and result.feedback.has_feedback():
            print(f"\n💡 Feedback:")
            
            if result.feedback.compilation_errors:
                print(f"\n   ❌ Compilation Errors:")
                for err in result.feedback.compilation_errors:
                    print(f"      {err}")
            
            if result.feedback.test_failures:
                print(f"\n   ❌ Test Failures:")
                for test in result.feedback.test_failures:
                    print(f"      Test #{test.test_number}: {test.description}")
                    if test.expected:
                        print(f"         Expected: {test.expected}")
                    if test.actual:
                        print(f"         Actual: {test.actual}")
            
            if result.feedback.behavioral_issues:
                print(f"\n   ⚠️  Behavioral Issues:")
                for issue in result.feedback.behavioral_issues:
                    print(f"      {issue}")
            
            if result.feedback.suggestions:
                print(f"\n   💡 Suggestions:")
                for sugg in result.feedback.suggestions:
                    print(f"      • {sugg}")
        
        # Error message
        if result.error_message:
            print(f"\n❌ Error: {result.error_message}")
        
        print(f"\n⏱️  Execution Time: {result.execution_time_ms:.1f} ms")
        print()
    
    def list_tasks(self, level: Optional[str] = None) -> List[dict]:
        """List available tasks"""
        tasks = self.agent.get_task_list(level=level)
        
        self.print_header(f"Available Tasks{f' (Level: {level})' if level else ''}")
        
        # Group by level
        by_level = {}
        for task in tasks:
            lvl = task['level']
            if lvl not in by_level:
                by_level[lvl] = []
            by_level[lvl].append(task)
        
        for lvl in sorted(by_level.keys()):
            print(f"\n📁 {lvl.upper()}:")
            for task in by_level[lvl]:
                print(f"   • {task['task_id']:<40} (Difficulty: {task['difficulty_score']})")
        
        print(f"\n{'='*70}")
        return tasks
    
    def read_code_from_file(self, filepath: str) -> str:
        """Read Verilog code from file"""
        try:
            with open(filepath, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return None
    
    def read_code_from_stdin(self) -> str:
        """Read multiline Verilog code from stdin"""
        print("📝 Enter your Verilog code (press Ctrl+D or type 'END' on a new line to finish):")
        print("-" * 70)
        
        lines = []
        try:
            while True:
                line = input()
                if line.strip() == 'END':
                    break
                lines.append(line)
        except EOFError:
            pass
        
        return '\n'.join(lines)
    
    def select_task(self, task_id: Optional[str] = None) -> bool:
        """Select a task to work on"""
        if not task_id:
            task_id = input("\n🎯 Enter task ID (e.g., level-0/mux2to1): ").strip()
        
        try:
            self.current_task = self.agent.get_task(task_id, include_testbench=False)
            self.print_task_info(self.current_task)
            return True
        except ValueError as e:
            print(f"❌ Error: {e}")
            return False
    
    def evaluate_code(
        self, 
        code: str, 
        validate_architecture: bool = True,
        use_llm_validation: bool = False,
        generate_feedback: bool = True
    ):
        """Evaluate Verilog code"""
        if not self.current_task:
            print("❌ No task selected! Use 'select' command first.")
            return
        
        print("\n⏳ Evaluating your code...")
        
        result = self.agent.evaluate_submission(
            task_id=self.current_task['task_id'],
            verilog_code=code,
            validate_architecture=validate_architecture,
            use_llm_validation=use_llm_validation,
            generate_feedback=generate_feedback
        )
        
        self.print_results(result)
    
    def run_interactive(self):
        """Run interactive REPL-style interface"""
        self.clear_screen()
        self.print_header("ArchXBench Interactive Evaluation")
        
        print("🚀 Welcome to ArchXBench!")
        print("\nCommands:")
        print("  list [level]          - List available tasks")
        print("  select <task_id>      - Select a task to work on")
        print("  show                  - Show current task details")
        print("  submit <file>         - Submit code from file")
        print("  submit                - Submit code from stdin")
        print("  help                  - Show this help")
        print("  quit                  - Exit")
        print()
        
        while True:
            try:
                cmd = input("📌 > ").strip()
                
                if not cmd:
                    continue
                
                parts = cmd.split(maxsplit=1)
                action = parts[0].lower()
                args = parts[1] if len(parts) > 1 else None
                
                if action in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                elif action == 'help':
                    print("\nCommands:")
                    print("  list [level]          - List available tasks")
                    print("  select <task_id>      - Select a task to work on")
                    print("  show                  - Show current task details")
                    print("  submit <file>         - Submit code from file")
                    print("  submit                - Submit code from stdin")
                    print("  help                  - Show this help")
                    print("  quit                  - Exit")
                
                elif action == 'list':
                    level = args.strip() if args else None
                    self.list_tasks(level=level)
                
                elif action == 'select':
                    if not args:
                        print("❌ Usage: select <task_id>")
                    else:
                        self.select_task(args.strip())
                
                elif action == 'show':
                    if self.current_task:
                        self.print_task_info(self.current_task)
                    else:
                        print("❌ No task selected! Use 'select' command first.")
                
                elif action == 'submit':
                    if not self.current_task:
                        print("❌ No task selected! Use 'select' command first.")
                        continue
                    
                    if args:
                        # Read from file
                        code = self.read_code_from_file(args.strip())
                        if code:
                            self.evaluate_code(code)
                    else:
                        # Read from stdin
                        code = self.read_code_from_stdin()
                        if code.strip():
                            self.evaluate_code(code)
                        else:
                            print("❌ No code provided!")
                
                else:
                    print(f"❌ Unknown command: {action}. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run_single_task(
        self, 
        task_id: str, 
        code_file: Optional[str] = None,
        validate_architecture: bool = True,
        use_llm_validation: bool = False
    ):
        """Evaluate a single task (non-interactive)"""
        # Select task
        if not self.select_task(task_id):
            return False
        
        # Get code
        if code_file:
            code = self.read_code_from_file(code_file)
            if not code:
                return False
        else:
            code = self.read_code_from_stdin()
        
        if not code.strip():
            print("❌ No code provided!")
            return False
        
        # Evaluate
        self.evaluate_code(code, validate_architecture, use_llm_validation)
        return True


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ArchXBench Interactive Evaluation - Solve RTL tasks manually",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m green_agent.interactive
  
  # Evaluate specific task from file
  python -m green_agent.interactive --task level-0/mux2to1 --file my_solution.v
  
  # List all tasks
  python -m green_agent.interactive --list
  
  # List tasks for specific level
  python -m green_agent.interactive --list --level level-0
        """
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available tasks and exit'
    )
    parser.add_argument(
        '--level',
        help='Filter tasks by level (use with --list)'
    )
    parser.add_argument(
        '--task',
        help='Task ID to evaluate (e.g., level-0/mux2to1)'
    )
    parser.add_argument(
        '--file',
        help='Verilog file to evaluate (use with --task)'
    )
    parser.add_argument(
        '--no-arch-validation',
        action='store_true',
        help='Disable architectural validation'
    )
    parser.add_argument(
        '--llm-validation',
        action='store_true',
        help='Enable LLM-based architectural validation (requires API key)'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        default=True,
        help='Use dynamic benchmark loading from GitHub (default)'
    )
    parser.add_argument(
        '--static',
        action='store_true',
        help='Use static local benchmarks'
    )
    parser.add_argument(
        '--benchmark-root',
        help='Path to local benchmark directory (for static mode)'
    )
    
    args = parser.parse_args()
    
    # Determine mode
    use_dynamic = not args.static
    benchmark_root = args.benchmark_root or os.environ.get("ARCHXBENCH_ROOT")
    
    # Create agent
    print("🔧 Initializing green agent...")
    agent = create_green_agent(
        benchmark_root=benchmark_root,
        use_dynamic_loader=use_dynamic
    )
    print(f"✅ Loaded {len(agent.tasks)} tasks")
    
    # Create session
    session = InteractiveSession(agent)
    
    # Handle different modes
    if args.list:
        # List mode
        session.list_tasks(level=args.level)
    
    elif args.task:
        # Single task evaluation mode
        session.run_single_task(
            task_id=args.task,
            code_file=args.file,
            validate_architecture=not args.no_arch_validation,
            use_llm_validation=args.llm_validation
        )
    
    else:
        # Interactive mode
        session.run_interactive()


if __name__ == "__main__":
    main()
