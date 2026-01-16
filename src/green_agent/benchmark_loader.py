"""
Dynamic Benchmark Loader for ArchXBench

Fetches benchmark tasks from the public ArchXBench repository instead of
using static local files. Supports caching and automatic updates.
"""

import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, List
import subprocess
from datetime import datetime, timedelta


class BenchmarkLoader:
    """
    Dynamically loads benchmark tasks from GitHub repository
    """
    
    # Public ArchXBench repository
    REPO_URL = "https://github.com/sureshpurini/ArchXBench.git"
    
    # Cache configuration
    DEFAULT_CACHE_DIR = Path.home() / ".archxbench" / "cache"
    CACHE_EXPIRY_HOURS = 24  # Refresh cache after 24 hours
    
    def __init__(
        self, 
        cache_dir: Optional[Path] = None,
        auto_update: bool = True,
        branch: str = "main"
    ):
        """
        Initialize benchmark loader
        
        Args:
            cache_dir: Directory to cache benchmarks (default: ~/.archxbench/cache)
            auto_update: Automatically update cache if expired
            branch: Git branch to fetch from
        """
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.auto_update = auto_update
        self.branch = branch
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.cache_dir / "metadata.json"
        
    def get_benchmark_root(self) -> Path:
        """
        Get the benchmark root directory, fetching from remote if needed
        
        Returns:
            Path to local benchmark directory
        """
        if self._is_cache_valid() and not self._should_update():
            print(f"Using cached benchmarks from {self.cache_dir / 'benchmarks'}")
            return self.cache_dir / "benchmarks"
        
        if self.auto_update:
            print("Fetching latest benchmarks from GitHub...")
            self._fetch_benchmarks()
            return self.cache_dir / "benchmarks"
        else:
            # Return cached version even if expired
            benchmark_path = self.cache_dir / "benchmarks"
            if benchmark_path.exists():
                return benchmark_path
            else:
                # Force fetch if no cache exists
                print("No cache found. Fetching benchmarks...")
                self._fetch_benchmarks()
                return self.cache_dir / "benchmarks"
    
    def _is_cache_valid(self) -> bool:
        """Check if cached benchmarks exist"""
        benchmark_path = self.cache_dir / "benchmarks"
        return benchmark_path.exists() and (benchmark_path / "level-0").exists()
    
    def _should_update(self) -> bool:
        """Check if cache should be updated based on age"""
        if not self.metadata_file.exists():
            return True
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            last_update = datetime.fromisoformat(metadata.get('last_update', '2000-01-01'))
            age = datetime.now() - last_update
            
            return age > timedelta(hours=self.CACHE_EXPIRY_HOURS)
        except Exception:
            return True
    
    def _fetch_benchmarks(self) -> None:
        """
        Fetch benchmarks from GitHub repository
        """
        benchmark_path = self.cache_dir / "benchmarks"
        
        # Create temporary directory for cloning
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "repo"
            
            try:
                # Clone repository (shallow clone for speed)
                print(f"  Cloning {self.REPO_URL}...")
                result = subprocess.run(
                    [
                        "git", "clone",
                        "--depth", "1",
                        "--branch", self.branch,
                        self.REPO_URL,
                        str(tmp_path)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode != 0:
                    error_msg = f"Git clone failed: {result.stderr}"
                    print(f"ERROR: {error_msg}")
                    print(f"  Command: git clone --depth 1 --branch {self.branch} {self.REPO_URL}")
                    print(f"  stdout: {result.stdout}")
                    print(f"  stderr: {result.stderr}")
                    raise RuntimeError(error_msg)
                
                print("  Clone successful")
                
                # Remove old benchmark cache if exists
                if benchmark_path.exists():
                    shutil.rmtree(benchmark_path)
                
                # Copy benchmark levels to cache
                benchmark_path.mkdir(parents=True, exist_ok=True)
                
                levels_copied = 0
                for level_dir in tmp_path.glob("level-*"):
                    if level_dir.is_dir():
                        dest = benchmark_path / level_dir.name
                        shutil.copytree(level_dir, dest)
                        levels_copied += 1
                
                print(f"  Copied {levels_copied} benchmark levels")
                
                # Save metadata
                metadata = {
                    'last_update': datetime.now().isoformat(),
                    'repo_url': self.REPO_URL,
                    'branch': self.branch,
                    'levels': levels_copied
                }
                
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print("✓ Benchmarks updated successfully")
                
            except subprocess.TimeoutExpired:
                raise RuntimeError("Git clone timed out after 5 minutes")
            except Exception as e:
                raise RuntimeError(f"Failed to fetch benchmarks: {str(e)}")
    
    def force_update(self) -> None:
        """Force update of cached benchmarks"""
        print("Forcing benchmark update...")
        self._fetch_benchmarks()
    
    def get_cache_info(self) -> Dict:
        """Get information about cached benchmarks"""
        if not self.metadata_file.exists():
            return {
                'cached': False,
                'message': 'No cache found'
            }
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            last_update = datetime.fromisoformat(metadata.get('last_update'))
            age = datetime.now() - last_update
            
            return {
                'cached': True,
                'last_update': metadata.get('last_update'),
                'age_hours': age.total_seconds() / 3600,
                'repo_url': metadata.get('repo_url'),
                'branch': metadata.get('branch'),
                'levels': metadata.get('levels'),
                'cache_dir': str(self.cache_dir / 'benchmarks'),
                'expired': age > timedelta(hours=self.CACHE_EXPIRY_HOURS)
            }
        except Exception as e:
            return {
                'cached': False,
                'error': str(e)
            }
    
    def clear_cache(self) -> None:
        """Clear cached benchmarks"""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print("✓ Cache cleared")


def get_benchmark_loader(
    cache_dir: Optional[Path] = None,
    auto_update: bool = True
) -> BenchmarkLoader:
    """
    Factory function to get benchmark loader instance
    
    Args:
        cache_dir: Custom cache directory (optional)
        auto_update: Enable automatic cache updates
    
    Returns:
        BenchmarkLoader instance
    """
    return BenchmarkLoader(
        cache_dir=cache_dir,
        auto_update=auto_update
    )
