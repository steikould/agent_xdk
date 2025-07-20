#!/usr/bin/env python3
"""
setup_testing_environment.py
============================

Python script to set up a unified testing environment for multi-agent system.
Handles dependency conflicts and creates a clean environment.
"""

import os
import subprocess
import sys
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

class TestingEnvironmentSetup:
    """Sets up unified testing environment for multi-agent system."""
    
    def __init__(self, project_root: Path = Path.current()):
        self.project_root = project_root
        self.venv_name = "venv_testing"
        self.venv_path = project_root / self.venv_name
        self.requirements_files = []
        self.all_requirements = defaultdict(list)
        
    def run(self):
        """Run the complete setup process."""
        print("🚀 Setting up unified testing environment for multi-agent system...")
        
        # Step 1: Create virtual environment
        self.create_virtual_environment()
        
        # Step 2: Collect all requirements
        self.collect_requirements()
        
        # Step 3: Resolve conflicts and create unified requirements
        self.create_unified_requirements()
        
        # Step 4: Install requirements
        self.install_requirements()
        
        # Step 5: Create necessary __init__.py files
        self.ensure_package_structure()
        
        # Step 6: Create or update setup.py
        self.create_setup_py()
        
        # Step 7: Install in development mode
        self.install_development_mode()
        
        # Step 8: Verify installation
        self.verify_installation()
        
        print("\n✅ Setup complete!")
        self.print_next_steps()
    
    def create_virtual_environment(self):
        """Create a new virtual environment."""
        print("\n📁 Creating virtual environment...")
        
        if self.venv_path.exists():
            response = input(f"Virtual environment {self.venv_name} already exists. Recreate? (y/N): ")
            if response.lower() != 'y':
                print("Using existing virtual environment.")
                return
            
            # Remove existing venv
            import shutil
            shutil.rmtree(self.venv_path)
        
        # Create new venv
        subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], check=True)
        print(f"✓ Created virtual environment: {self.venv_name}")
    
    def get_pip_command(self) -> List[str]:
        """Get the pip command for the virtual environment."""
        if sys.platform == "win32":
            pip_path = self.venv_path / "Scripts" / "pip.exe"
        else:
            pip_path = self.venv_path / "bin" / "pip"
        return [str(pip_path)]
    
    def get_python_command(self) -> List[str]:
        """Get the python command for the virtual environment."""
        if sys.platform == "win32":
            python_path = self.venv_path / "Scripts" / "python.exe"
        else:
            python_path = self.venv_path / "bin" / "python"
        return [str(python_path)]
    
    def collect_requirements(self):
        """Collect all requirements from agent directories."""
        print("\n📋 Collecting requirements from all agents...")
        
        # Common requirement file patterns
        requirement_patterns = [
            "**/requirements*.txt",
            "**/requirements/*.txt",
            "agents/*/requirements*.txt",
        ]
        
        for pattern in requirement_patterns:
            for req_file in self.project_root.glob(pattern):
                # Skip files in virtual environments
                if self.venv_name in str(req_file) or "venv" in str(req_file):
                    continue
                
                self.requirements_files.append(req_file)
                print(f"  Found: {req_file.relative_to(self.project_root)}")
                
                # Parse requirements
                self.parse_requirements_file(req_file)
        
        # Also check for pyproject.toml files
        for pyproject in self.project_root.glob("**/pyproject.toml"):
            if self.venv_name not in str(pyproject):
                self.parse_pyproject_toml(pyproject)
    
    def parse_requirements_file(self, req_file: Path):
        """Parse a requirements.txt file."""
        try:
            with open(req_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('-'):
                        self.parse_requirement_line(line, str(req_file))
        except Exception as e:
            print(f"  ⚠️  Error reading {req_file}: {e}")
    
    def parse_requirement_line(self, line: str, source: str):
        """Parse a single requirement line."""
        # Handle different requirement formats
        # package==1.0.0
        # package>=1.0.0
        # package~=1.0.0
        # package[extra]>=1.0.0
        
        match = re.match(r'^([a-zA-Z0-9\-_]+)(\[.*?\])?([=<>~!]+.*)?$', line)
        if match:
            package = match.group(1).lower()
            extras = match.group(2) or ''
            version_spec = match.group(3) or ''
            
            self.all_requirements[package].append({
                'spec': f"{package}{extras}{version_spec}",
                'version': version_spec,
                'source': source
            })
    
    def parse_pyproject_toml(self, pyproject_path: Path):
        """Parse dependencies from pyproject.toml."""
        try:
            import toml
        except ImportError:
            print(f"  ⚠️  toml not installed, skipping {pyproject_path}")
            return
        
        try:
            with open(pyproject_path, 'r') as f:
                data = toml.load(f)
            
            # Check for poetry dependencies
            if 'tool' in data and 'poetry' in data['tool']:
                deps = data['tool']['poetry'].get('dependencies', {})
                for package, version in deps.items():
                    if package.lower() != 'python':
                        version_spec = self.parse_poetry_version(version)
                        self.all_requirements[package.lower()].append({
                            'spec': f"{package}{version_spec}",
                            'version': version_spec,
                            'source': str(pyproject_path)
                        })
            
            # Check for setuptools dependencies
            elif 'project' in data:
                deps = data['project'].get('dependencies', [])
                for dep in deps:
                    self.parse_requirement_line(dep, str(pyproject_path))
                    
        except Exception as e:
            print(f"  ⚠️  Error parsing {pyproject_path}: {e}")
    
    def parse_poetry_version(self, version) -> str:
        """Convert poetry version spec to pip format."""
        if isinstance(version, dict):
            if 'version' in version:
                version = version['version']
            else:
                return ""
        
        if version.startswith('^'):
            # ^1.2.3 -> >=1.2.3,<2.0.0
            return f">={version[1:]}"
        elif version.startswith('~'):
            # ~1.2.3 -> >=1.2.3,<1.3.0
            return f"~={version[1:]}"
        else:
            return f"=={version}"
    
    def create_unified_requirements(self):
        """Create unified requirements file resolving conflicts."""
        print("\n🔧 Creating unified requirements file...")
        
        # Base testing requirements
        base_requirements = {
            'pytest': '>=7.0.0',
            'pytest-asyncio': '>=0.21.0',
            'pytest-cov': '>=4.0.0',
            'pytest-mock': '>=3.10.0',
            'pandas': '>=2.0.0',
            'numpy': '>=1.24.0',
            'pyyaml': '>=6.0',
            'python-dotenv': '>=1.0.0',
            'pydantic': '>=2.0.0',
        }
        
        # Add base requirements
        for package, version in base_requirements.items():
            if package not in self.all_requirements:
                self.all_requirements[package].append({
                    'spec': f"{package}{version}",
                    'version': version,
                    'source': 'base_testing'
                })
        
        # Resolve conflicts and write unified requirements
        unified_reqs = []
        conflicts = []
        
        for package, specs in self.all_requirements.items():
            if len(specs) == 1:
                unified_reqs.append(specs[0]['spec'])
            else:
                # Multiple versions specified - need to resolve
                resolved = self.resolve_version_conflict(package, specs)
                if resolved:
                    unified_reqs.append(resolved)
                else:
                    conflicts.append((package, specs))
        
        # Write unified requirements
        unified_path = self.project_root / "requirements_unified.txt"
        with open(unified_path, 'w') as f:
            f.write("# Unified requirements for all agents\n")
            f.write("# Generated by setup_testing_environment.py\n\n")
            
            for req in sorted(unified_reqs):
                f.write(f"{req}\n")
        
        print(f"✓ Created unified requirements: {unified_path}")
        
        # Report conflicts
        if conflicts:
            print("\n⚠️  Version conflicts detected:")
            for package, specs in conflicts:
                print(f"\n  {package}:")
                for spec in specs:
                    print(f"    - {spec['spec']} (from {Path(spec['source']).name})")
    
    def resolve_version_conflict(self, package: str, specs: List[Dict]) -> str:
        """Attempt to resolve version conflicts."""
        # Simple resolution strategy - take the most restrictive version
        # In practice, you might want more sophisticated resolution
        
        versions = [s['version'] for s in specs if s['version']]
        if not versions:
            return package
        
        # For now, just take the first specific version
        for spec in specs:
            if spec['version'] and '==' in spec['version']:
                return spec['spec']
        
        # Otherwise, take the first one
        return specs[0]['spec']
    
    def install_requirements(self):
        """Install all requirements."""
        print("\n📦 Installing requirements...")
        
        # Upgrade pip first
        pip_cmd = self.get_pip_command()
        subprocess.run(pip_cmd + ["install", "--upgrade", "pip"], check=True)
        
        # Install unified requirements
        unified_path = self.project_root / "requirements_unified.txt"
        if unified_path.exists():
            subprocess.run(pip_cmd + ["install", "-r", str(unified_path)], check=True)
            print("✓ Installed unified requirements")
    
    def ensure_package_structure(self):
        """Ensure all directories have __init__.py files."""
        print("\n📂 Ensuring proper package structure...")
        
        # Directories that should be packages
        package_dirs = [
            "agents",
            "tests",
        ]
        
        # Also find all subdirectories under agents
        agents_dir = self.project_root / "agents"
        if agents_dir.exists():
            for subdir in agents_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    package_dirs.append(f"agents/{subdir.name}")
        
        for dir_path in package_dirs:
            dir_full_path = self.project_root / dir_path
            if dir_full_path.exists():
                init_file = dir_full_path / "__init__.py"
                if not init_file.exists():
                    init_file.write_text("# Auto-generated by setup_testing_environment.py\n")
                    print(f"  Created: {dir_path}/__init__.py")
    
    def create_setup_py(self):
        """Create or update setup.py for development installation."""
        print("\n📄 Creating setup.py...")
        
        setup_py_path = self.project_root / "setup.py"
        
        if setup_py_path.exists():
            print("  setup.py already exists, skipping...")
            return
        
        setup_content = '''"""
Setup configuration for multi-agent system.
Auto-generated by setup_testing_environment.py
"""

from setuptools import setup, find_packages

# Read requirements
with open("requirements_unified.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="multi-agent-system",
    version="0.1.0",
    description="Multi-agent system for pump analysis",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
        ],
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "ipython>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-tests=tests.run_tests:main",
        ],
    },
)
'''
        
        setup_py_path.write_text(setup_content)
        print("✓ Created setup.py")
    
    def install_development_mode(self):
        """Install the package in development mode."""
        print("\n🔗 Installing package in development mode...")
        
        pip_cmd = self.get_pip_command()
        try:
            subprocess.run(pip_cmd + ["install", "-e", "."], check=True)
            print("✓ Installed in development mode")
        except subprocess.CalledProcessError:
            print("⚠️  Could not install in development mode (this is okay if no setup.py)")
    
    def verify_installation(self):
        """Verify that key packages can be imported."""
        print("\n✔️  Verifying installation...")
        
        python_cmd = self.get_python_command()
        
        verification_script = '''
import sys
import importlib

packages_to_check = [
    ("pandas", True),
    ("numpy", True),
    ("pytest", True),
    ("yaml", True),
    ("pydantic", True),
    ("asyncio", True),
]

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print()

success_count = 0
for package, required in packages_to_check:
    try:
        if package == "yaml":
            importlib.import_module("yaml")
            print(f"✓ {package}: installed")
        else:
            mod = importlib.import_module(package)
            version = getattr(mod, "__version__", "unknown")
            print(f"✓ {package}: {version}")
        success_count += 1
    except ImportError:
        if required:
            print(f"✗ {package}: NOT INSTALLED (required)")
        else:
            print(f"- {package}: not installed (optional)")

print(f"\\nSuccessfully imported {success_count}/{len(packages_to_check)} packages")

# Try to import agents
print("\\nChecking agent imports...")
try:
    import agents
    print("✓ Can import agents package")
    
    # Try specific agents
    agent_modules = ["data_quality_agent", "power_calculation_agent", "statistical_analysis_agent"]
    for agent in agent_modules:
        try:
            importlib.import_module(f"agents.{agent}")
            print(f"  ✓ agents.{agent}")
        except ImportError as e:
            print(f"  ✗ agents.{agent}: {str(e)}")
except ImportError:
    print("✗ Cannot import agents package - make sure to run from project root")
'''
        
        result = subprocess.run(
            python_cmd + ["-c", verification_script],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    
    def print_next_steps(self):
        """Print next steps for the user."""
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        
        if sys.platform == "win32":
            activate_cmd = f"{self.venv_name}\\Scripts\\activate"
        else:
            activate_cmd = f"source {self.venv_name}/bin/activate"
        
        print(f"""
1. Activate the virtual environment:
   {activate_cmd}

2. Run the tests:
   python tests/run_tests.py --level basic

3. Run specific agent tests:
   python tests/run_tests.py --agents DataQualityAgent --level all

4. Run with enhancement iterations:
   python tests/run_tests.py --level all --enhance

5. View test results:
   cat test_results.json

Troubleshooting:
- If imports still fail, check that __init__.py files exist in all directories
- Make sure to run tests from the project root directory
- Check that agent class names match what's in the test files
""")


def main():
    """Main entry point."""
    setup = TestingEnvironmentSetup()
    
    try:
        setup.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()