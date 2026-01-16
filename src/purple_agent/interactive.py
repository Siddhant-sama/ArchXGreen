#!/usr/bin/env python3
"""
Interactive CLI for ArchXBench Purple Agent

Provides an intuitive interface to:
- Select LLM provider and model
- Browse available tasks and levels
- Run evaluations with visual feedback
"""

import os
import sys
from typing import Optional, List
import requests

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purple_agent.agent import (
    create_openai_agent,
    create_anthropic_agent,
    create_gemini_agent,
    create_openrouter_agent,
    create_local_agent,
    ArchXBenchPurpleAgent
)


# ========== Configuration ==========

PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "models": ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
        "default": "gpt-4",
        "env_key": "OPENAI_API_KEY",
        "key_format": "sk-..."
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "models": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022"
        ],
        "default": "claude-3-sonnet-20240229",
        "env_key": "ANTHROPIC_API_KEY",
        "key_format": "sk-ant-..."
    },
    "gemini": {
        "name": "Google Gemini",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
        "default": "gemini-1.5-pro",
        "env_key": "GEMINI_API_KEY",
        "key_format": "..."
    },
    "openrouter": {
        "name": "OpenRouter",
        "models": [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4-turbo",
            "google/gemini-pro",
            "meta-llama/llama-3.1-70b-instruct"
        ],
        "default": "anthropic/claude-3.5-sonnet",
        "env_key": "OPENROUTER_API_KEY",
        "key_format": "sk-or-..."
    },
    "local": {
        "name": "Local LLM (Ollama)",
        "models": ["codellama", "deepseek-coder", "llama3", "mistral"],
        "default": "codellama",
        "env_key": None,
        "key_format": None
    }
}


# ========== Helper Functions ==========

def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text: str):
    """Print a section divider"""
    print(f"\n--- {text} ---")


def get_choice(prompt: str, options: list, allow_back: bool = True) -> Optional[str]:
    """Get user choice from a list of options"""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    if allow_back:
        print(f"  0. Back/Cancel")
    
    while True:
        try:
            choice = input("\nEnter choice: ").strip()
            if not choice:
                continue
            
            choice_num = int(choice)
            if choice_num == 0 and allow_back:
                return None
            if 1 <= choice_num <= len(options):
                return options[choice_num - 1]
            print(f"Invalid choice. Enter 1-{len(options)}" + (" or 0" if allow_back else ""))
        except (ValueError, KeyboardInterrupt):
            if allow_back:
                return None
            print("Invalid input. Please enter a number.")


def get_input(prompt: str, default: Optional[str] = None) -> str:
    """Get text input from user"""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    
    value = input(prompt).strip()
    return value if value else (default or "")


def confirm(prompt: str, default: bool = True) -> bool:
    """Ask yes/no confirmation"""
    suffix = " [Y/n]: " if default else " [y/N]: "
    response = input(prompt + suffix).strip().lower()
    
    if not response:
        return default
    return response in ['y', 'yes']


# ========== Main Interactive Flow ==========

class PurpleAgentCLI:
    """Interactive CLI for Purple Agent"""
    
    def __init__(self):
        self.green_agent_url = None
        self.provider = None
        self.model = None
        self.api_key = None
        self.agent = None
        self.available_tasks = []
        self.available_levels = []
    
    def run(self):
        """Main interactive loop"""
        print_header("🟣 ArchXBench Purple Agent - Interactive Mode")
        print("\nAutonomous LLM-powered Verilog RTL generator")
        print("Solves benchmark tasks with iterative refinement\n")
        
        # Step 1: Configure green agent URL
        if not self.configure_green_agent():
            return
        
        # Step 2: Select LLM provider
        if not self.select_provider():
            return
        
        # Step 3: Select model
        if not self.select_model():
            return
        
        # Step 4: Configure API key (if needed)
        if not self.configure_api_key():
            return
        
        # Step 5: Create agent
        if not self.create_agent():
            return
        
        # Step 6: Fetch tasks from green agent
        if not self.fetch_tasks():
            return
        
        # Main menu loop
        while True:
            choice = self.main_menu()
            if choice is None or choice == "exit":
                break
            
            if choice == "list_levels":
                self.show_levels()
            elif choice == "list_tasks":
                self.show_tasks()
            elif choice == "run_level":
                self.run_by_level()
            elif choice == "run_tasks":
                self.run_specific_tasks()
            elif choice == "run_all":
                self.run_all_tasks()
            elif choice == "change_provider":
                if self.select_provider() and self.select_model() and self.configure_api_key():
                    self.create_agent()
        
        print("\n👋 Goodbye!\n")
    
    def configure_green_agent(self) -> bool:
        """Configure green agent URL"""
        print_section("Green Agent Configuration")
        
        default_url = os.environ.get("GREEN_AGENT_URL", "http://localhost:8000")
        self.green_agent_url = get_input(
            "Enter green agent URL",
            default=default_url
        )
        
        # Test connection
        print(f"\nTesting connection to {self.green_agent_url}...")
        try:
            response = requests.get(f"{self.green_agent_url}/health", timeout=5)
            response.raise_for_status()
            data = response.json()
            print(f"✓ Connected to {data['benchmark']} v{data['version']}")
            print(f"  Tasks available: {data['total_tasks']}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect: {e}")
            print("\nMake sure the green agent is running:")
            print("  docker compose up -d")
            print("  OR")
            print("  python -m green_agent.server")
            return False
    
    def select_provider(self) -> bool:
        """Select LLM provider"""
        print_section("Select LLM Provider")
        
        provider_names = [f"{info['name']}" for info in PROVIDERS.values()]
        provider_keys = list(PROVIDERS.keys())
        
        selected = get_choice("Choose LLM provider:", provider_names, allow_back=False)
        if not selected:
            return False
        
        idx = provider_names.index(selected)
        self.provider = provider_keys[idx]
        
        print(f"\n✓ Selected: {PROVIDERS[self.provider]['name']}")
        return True
    
    def select_model(self) -> bool:
        """Select model for the provider"""
        print_section(f"Select Model ({PROVIDERS[self.provider]['name']})")
        
        models = PROVIDERS[self.provider]['models']
        default_model = PROVIDERS[self.provider]['default']
        
        print(f"\nAvailable models:")
        for i, model in enumerate(models, 1):
            marker = " (default)" if model == default_model else ""
            print(f"  {i}. {model}{marker}")
        
        choice = get_input("\nEnter model number or name", default=str(1))
        
        # Parse choice
        try:
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(models):
                self.model = models[model_idx]
            else:
                self.model = default_model
        except ValueError:
            # User entered model name directly
            if choice in models:
                self.model = choice
            else:
                self.model = default_model
        
        print(f"\n✓ Selected model: {self.model}")
        return True
    
    def configure_api_key(self) -> bool:
        """Configure API key if needed"""
        provider_info = PROVIDERS[self.provider]
        
        # Local LLM doesn't need API key
        if not provider_info['env_key']:
            print("\n✓ No API key required for local LLM")
            return True
        
        print_section("API Key Configuration")
        
        env_key = provider_info['env_key']
        key_format = provider_info['key_format']
        
        # Check if key already in environment
        self.api_key = os.environ.get(env_key)
        if self.api_key:
            masked = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else "***"
            print(f"\n✓ Found API key in environment: {masked}")
            use_env = confirm("Use this key?", default=True)
            if use_env:
                return True
        
        # Prompt for key
        print(f"\nAPI key format: {key_format}")
        print(f"Get your key from: {self._get_key_url(self.provider)}")
        
        self.api_key = get_input(f"\nEnter {provider_info['name']} API key")
        
        if not self.api_key:
            print("✗ API key required")
            return False
        
        # Save to environment for this session
        os.environ[env_key] = self.api_key
        print(f"✓ API key configured for this session")
        
        return True
    
    def _get_key_url(self, provider: str) -> str:
        """Get URL for obtaining API key"""
        urls = {
            "openai": "https://platform.openai.com/api-keys",
            "anthropic": "https://console.anthropic.com/settings/keys",
            "gemini": "https://makersuite.google.com/app/apikey",
            "openrouter": "https://openrouter.ai/keys"
        }
        return urls.get(provider, "provider website")
    
    def create_agent(self) -> bool:
        """Create the purple agent instance"""
        print_section("Initializing Agent")
        
        try:
            if self.provider == "openai":
                self.agent = create_openai_agent(
                    self.green_agent_url,
                    api_key=self.api_key,
                    model=self.model,
                    verbose=True
                )
            elif self.provider == "anthropic":
                self.agent = create_anthropic_agent(
                    self.green_agent_url,
                    api_key=self.api_key,
                    model=self.model,
                    verbose=True
                )
            elif self.provider == "gemini":
                self.agent = create_gemini_agent(
                    self.green_agent_url,
                    api_key=self.api_key,
                    model=self.model,
                    verbose=True
                )
            elif self.provider == "openrouter":
                self.agent = create_openrouter_agent(
                    self.green_agent_url,
                    api_key=self.api_key,
                    model=self.model,
                    verbose=True
                )
            elif self.provider == "local":
                self.agent = create_local_agent(
                    self.green_agent_url,
                    model=self.model,
                    verbose=True
                )
            
            print(f"✓ Agent created successfully")
            print(f"  Provider: {PROVIDERS[self.provider]['name']}")
            print(f"  Model: {self.model}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to create agent: {e}")
            return False
    
    def fetch_tasks(self) -> bool:
        """Fetch available tasks from green agent"""
        print_section("Fetching Tasks")
        
        try:
            print("Querying green agent for available tasks...")
            self.available_tasks = self.agent.get_available_tasks()
            
            # Extract unique levels
            self.available_levels = sorted(set(task['level'] for task in self.available_tasks))
            
            print(f"✓ Found {len(self.available_tasks)} tasks across {len(self.available_levels)} levels")
            return True
            
        except Exception as e:
            print(f"✗ Failed to fetch tasks: {e}")
            return False
    
    def main_menu(self) -> Optional[str]:
        """Show main menu"""
        print_header("Main Menu")
        
        options = [
            ("list_levels", "📊 List difficulty levels"),
            ("list_tasks", "📋 List all tasks"),
            ("run_level", "🎯 Run tasks by level"),
            ("run_tasks", "✅ Run specific tasks"),
            ("run_all", "🚀 Run all tasks"),
            ("change_provider", "🔄 Change LLM provider/model"),
            ("exit", "🚪 Exit")
        ]
        
        print()
        for i, (key, label) in enumerate(options, 1):
            print(f"  {i}. {label}")
        
        while True:
            try:
                choice = input("\nEnter choice (1-7): ").strip()
                if not choice:
                    continue
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(options):
                    return options[choice_num - 1][0]
                print(f"Invalid choice. Enter 1-{len(options)}")
            except (ValueError, KeyboardInterrupt):
                print("\n")
                return "exit"
    
    def show_levels(self):
        """Display available levels with task counts"""
        print_header("Difficulty Levels")
        
        for level in self.available_levels:
            tasks_in_level = [t for t in self.available_tasks if t['level'] == level]
            print(f"\n  {level}: {len(tasks_in_level)} tasks")
            
            # Show first 3 tasks as examples
            for task in tasks_in_level[:3]:
                print(f"    - {task['task_id']}")
            if len(tasks_in_level) > 3:
                print(f"    ... and {len(tasks_in_level) - 3} more")
    
    def show_tasks(self):
        """Display all tasks"""
        print_header("All Tasks")
        
        for level in self.available_levels:
            tasks_in_level = [t for t in self.available_tasks if t['level'] == level]
            print(f"\n{level} ({len(tasks_in_level)} tasks):")
            for task in tasks_in_level:
                print(f"  - {task['task_id']}")
    
    def run_by_level(self):
        """Run all tasks in a selected level"""
        print_header("Run Tasks by Level")
        
        level = get_choice("Select level:", self.available_levels)
        if not level:
            return
        
        tasks_in_level = [t for t in self.available_tasks if t['level'] == level]
        
        print(f"\nWill run {len(tasks_in_level)} tasks in {level}")
        if not confirm("Continue?"):
            return
        
        results = self.agent.run_benchmark(levels=[level])
        self._show_results(results)
    
    def run_specific_tasks(self):
        """Run specific selected tasks"""
        print_header("Run Specific Tasks")
        
        print("\nEnter task IDs (comma-separated):")
        print("Example: level-0/mux2to1, level-0/mux4to1")
        
        task_input = input("\nTask IDs: ").strip()
        if not task_input:
            return
        
        task_ids = [t.strip() for t in task_input.split(',')]
        
        print(f"\nWill run {len(task_ids)} tasks")
        if not confirm("Continue?"):
            return
        
        results = self.agent.run_benchmark(task_ids=task_ids)
        self._show_results(results)
    
    def run_all_tasks(self):
        """Run all available tasks"""
        print_header("Run All Tasks")
        
        print(f"\n⚠️  This will run ALL {len(self.available_tasks)} tasks")
        print("This may take a long time and use significant API credits")
        
        if not confirm("Are you sure?", default=False):
            return
        
        results = self.agent.run_benchmark()
        self._show_results(results)
    
    def _show_results(self, results):
        """Display benchmark results"""
        print_header("Results Summary")
        
        print(f"\n  Total Tasks:    {results.total_tasks}")
        print(f"  Passed:         {results.passed}")
        print(f"  Failed:         {results.failed}")
        print(f"  Pass Rate:      {results.pass_rate*100:.1f}%")
        print(f"  Execution Time: {results.execution_time_sec:.1f}s")
        
        # Show failed tasks
        failed_tasks = [r for r in results.results if not r.success]
        if failed_tasks:
            print(f"\n  Failed tasks:")
            for result in failed_tasks:
                print(f"    - {result.task_id} ({result.iterations} iterations)")


def main():
    """Entry point"""
    try:
        cli = PurpleAgentCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
