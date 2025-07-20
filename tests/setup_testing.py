#!/usr/bin/env python3
"""
Simple setup script for testing environment
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return success status."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and check:
            print(f"Warning: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False

def main():
    print("Setting up testing environment...")
    
    # Determine Python command
    python_cmd = sys.executable
    print(f"Using Python: {python_cmd}")
    
    # Create virtual environment
    venv_name = "venv_testing"
    print(f"\n1. Creating virtual environment: {venv_name}")
    if not run_command([python_cmd, "-m", "venv", venv_name]):
        print("Failed to create virtual environment")
        return 1
    
    # Determine pip path
    if sys.platform == "win32":
        pip_path = os.path.join(venv_name, "Scripts", "pip")
        python_venv = os.path.join(venv_name, "Scripts", "python")
    else:
        pip_path = os.path.join(venv_name, "bin", "pip")
        python_venv = os.path.join(venv_name, "bin", "python")
    
    # Upgrade pip
    print("\n2. Upgrading pip...")
    run_command([python_venv, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Create base requirements file
    print("\n3. Creating base requirements...")
    base_requirements = """# Base testing requirements
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
pandas>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
python-dotenv>=1.0.0
pydantic>=2.0.0
aiofiles>=23.0.0
"""
    
    with open("requirements_base.txt", "w") as f:
        f.write(base_requirements)
    
    # Install base requirements
    print("\n4. Installing base requirements...")
    run_command([pip_path, "install", "-r", "requirements_base.txt"])
    
    # Find and install agent requirements
    print("\n5. Looking for agent requirement files...")
    
    req_files = []
    # Look for requirements files
    for pattern in ["agents/*/requirements.txt", "*/requirements.txt", "requirements.txt"]:
        for req_file in Path(".").glob(pattern):
            if "venv" not in str(req_file):
                req_files.append(req_file)
                print(f"   Found: {req_file}")
    
    # Install each requirements file
    for req_file in req_files:
        print(f"\n   Installing from {req_file}...")
        run_command([pip_path, "install", "-r", str(req_file)], check=False)
    
    # Create __init__.py files
    print("\n6. Creating __init__.py files...")
    dirs_to_init = ["agents", "tests"]
    
    # Also check for subdirectories in agents
    agents_dir = Path("agents")
    if agents_dir.exists():
        for subdir in agents_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                dirs_to_init.append(str(subdir))
    
    for dir_path in dirs_to_init:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            init_file = path / "__init__.py"
            if not init_file.exists():
                init_file.write_text("# Auto-generated\n")
                print(f"   Created: {init_file}")
    
    # Create simple setup.py if needed
    if not Path("setup.py").exists():
        print("\n7. Creating setup.py...")
        setup_content = """from setuptools import setup, find_packages

setup(
    name="multi-agent-system",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
)
"""
        with open("setup.py", "w") as f:
            f.write(setup_content)
    
    # Install in development mode
    print("\n8. Installing package in development mode...")
    run_command([pip_path, "install", "-e", "."], check=False)
    
    # Print summary
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print(f"\nTo activate the environment:")
    if sys.platform == "win32":
        print(f"   {venv_name}\\Scripts\\activate")
    else:
        print(f"   source {venv_name}/bin/activate")
    
    print(f"\nTo run tests:")
    print(f"   python tests/run_tests.py --level basic")
    
    print(f"\nInstalled packages:")
    run_command([pip_path, "list"])
    
    return 0

if __name__ == "__main__":
    sys.exit(main())