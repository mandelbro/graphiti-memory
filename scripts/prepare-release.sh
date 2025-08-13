#!/bin/bash

# Graphiti MCP Server Release Preparation Script
# This script helps prepare a new release by updating version numbers and creating proper tags

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -f "src/graphiti_mcp_server.py" ]]; then
    print_error "This script must be run from the root of the montesmakes.graphiti-memory project"
    exit 1
fi

# Check if git working directory is clean
if [[ -n $(git status --porcelain) ]]; then
    print_error "Git working directory is not clean. Please commit or stash changes first."
    git status --short
    exit 1
fi

# Get current version from pyproject.toml
current_version=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
print_info "Current version: $current_version"

# Ask for new version
echo
read -p "Enter new version (e.g., 0.4.1, 0.5.0, 1.0.0): " new_version

# Validate version format (basic semantic versioning check)
if [[ ! $new_version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    print_error "Invalid version format. Please use semantic versioning (e.g., 1.2.3)"
    exit 1
fi

# Confirm the update
echo
print_warning "This will update the version from $current_version to $new_version"
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Cancelled by user"
    exit 0
fi

# Update version in pyproject.toml
print_info "Updating version in pyproject.toml..."
sed -i.bak "s/version = \"$current_version\"/version = \"$new_version\"/" pyproject.toml
rm pyproject.toml.bak

# Update version in src/__init__.py
print_info "Updating version in src/__init__.py..."
sed -i.bak "s/__version__ = \"$current_version\"/__version__ = \"$new_version\"/" src/__init__.py
rm src/__init__.py.bak

print_success "Version updated to $new_version"

# Run tests to make sure everything still works
print_info "Running tests to verify the changes..."
if command -v uv >/dev/null 2>&1; then
    uv run pytest --no-cov || {
        print_error "Tests failed! Please fix issues before releasing."
        exit 1
    }
else
    print_warning "uv not found, skipping tests"
fi

# Build the package to verify it can be built
print_info "Building package to verify it can be built..."
if command -v uv >/dev/null 2>&1; then
    uv build || {
        print_error "Build failed! Please fix issues before releasing."
        exit 1
    }
    print_success "Package built successfully"
else
    print_warning "uv not found, skipping build verification"
fi

# Create git commit
print_info "Creating git commit..."
git add pyproject.toml src/__init__.py
git commit -m "Bump version to $new_version"

# Create git tag
print_info "Creating git tag v$new_version..."
git tag -a "v$new_version" -m "Release version $new_version"

print_success "Release preparation complete!"
echo
print_info "Next steps:"
echo "  1. Push the changes: git push origin main"
echo "  2. Push the tag: git push origin v$new_version"
echo "  3. Create a GitHub release at: https://github.com/mandelbro/graphiti-memory/releases/new"
echo "     - Use tag: v$new_version"
echo "     - Title: Release v$new_version"
echo "     - Add release notes describing changes"
echo
print_info "The GitHub Action will automatically publish to PyPI when you create the GitHub release."
