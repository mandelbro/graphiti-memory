# PyPI Publishing Setup with Trusted Publishing

This guide walks you through setting up automated PyPI publishing for the Graphiti MCP Server using GitHub Actions and PyPI's trusted publishing feature.

## Overview

We use **trusted publishing** which is the modern, secure way to publish to PyPI without storing API tokens. It uses OpenID Connect (OIDC) to authenticate GitHub Actions with PyPI.

## Benefits of Trusted Publishing

- ✅ **No API tokens to manage** - No secrets to store or rotate
- ✅ **More secure** - Uses short-lived tokens from GitHub's OIDC provider
- ✅ **Better audit trail** - PyPI knows exactly which GitHub workflow published each release
- ✅ **Easier setup** - No need to generate and store PyPI tokens

## Setup Steps

### 1. Configure PyPI Trusted Publishing

#### For New Packages (First Time Publishing)

1. **Create a PyPI account** at https://pypi.org if you don't have one
2. **Go to your PyPI account settings**: https://pypi.org/manage/account/
3. **Navigate to "Trusted publishers"** section
4. **Add a new trusted publisher** with these details:
   - **Repository owner**: `mandelbro` (or your GitHub username/org)
   - **Repository name**: `graphiti-memory` (or your repo name)
   - **Workflow name**: `ci-cd.yml`
   - **Environment name**: `pypi`

#### For Existing Packages

1. Go to your project page on PyPI: https://pypi.org/project/montesmakes.graphiti-memory/
2. Go to **Settings** → **Trusted Publishers**
3. **Add a trusted publisher** with the same details as above

### 2. Configure TestPyPI (Optional but Recommended)

For testing releases before publishing to the main PyPI:

1. **Create a TestPyPI account** at https://test.pypi.org
2. **Add trusted publisher** for TestPyPI:
   - Repository owner: `mandelbro`
   - Repository name: `graphiti-memory`
   - Workflow name: `ci-cd.yml`
   - Environment name: `testpypi`

### 3. Create GitHub Environments

Create protected environments in your GitHub repository:

1. **Go to your repository** → **Settings** → **Environments**
2. **Create `pypi` environment**:
   - Add protection rules (require reviews for production)
   - Set deployment branch rules (only `main` branch)
3. **Create `testpypi` environment**:
   - Less restrictive rules for testing

## Workflow Explanation

Our CI/CD workflow (`.github/workflows/ci-cd.yml`) includes:

### Jobs Overview

1. **`test`** - Runs tests across Python versions 3.10-3.12
2. **`build`** - Builds the package and uploads artifacts
3. **`publish-test-pypi`** - Publishes to TestPyPI on `main` branch pushes
4. **`publish-pypi`** - Publishes to PyPI only on GitHub releases

### Security Features

- **Separated build/publish jobs** - Build happens in unrestricted environment
- **Restricted publish permissions** - Only `id-token: write` permission for publishing
- **Environment protection** - Publishing jobs run in protected environments
- **Attestation generation** - Creates signed attestations for all packages

## Publishing Process

### For Development/Testing (TestPyPI)

1. **Push to `main` branch** - Automatically triggers TestPyPI publishing
2. **Check TestPyPI**: https://test.pypi.org/project/montesmakes.graphiti-memory/
3. **Test installation**:
   ```bash
   uvx --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ montesmakes.graphiti-memory
   ```

### For Production Releases (PyPI)

1. **Create a GitHub release**:
   - Go to **Releases** → **Create a new release**
   - Create a new tag (e.g., `v0.4.0`)
   - Write release notes
   - Publish the release

2. **Automatic publishing** - The workflow automatically:
   - Builds the package
   - Publishes to PyPI
   - Creates signed attestations

3. **Verify publication**:
   - Check PyPI: https://pypi.org/project/montesmakes.graphiti-memory/
   - Test installation: `uvx montesmakes.graphiti-memory`

## Version Management

### Automatic Version Updates

Update the version in `pyproject.toml` before creating releases:

```toml
[project]
name = "montesmakes.graphiti-memory"
version = "0.5.0"  # Update this
```

### Recommended Versioning

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR** (1.0.0) - Breaking changes
- **MINOR** (0.5.0) - New features, backwards compatible
- **PATCH** (0.4.1) - Bug fixes, backwards compatible

## Manual Testing

Use the **Manual Package Test** workflow:

1. **Go to Actions** → **Manual Package Test**
2. **Run workflow** with options:
   - Test with TestPyPI or PyPI
   - Choose Python version
3. **Review results** to ensure package works correctly

## Troubleshooting

### Common Issues

#### 1. "Package already exists" error
- **Cause**: Version already published
- **Solution**: Update version in `pyproject.toml`

#### 2. "Trusted publisher not configured" error
- **Cause**: PyPI trusted publisher not set up correctly
- **Solution**: Double-check repository name, owner, and workflow details

#### 3. "Permission denied" error
- **Cause**: Missing `id-token: write` permission
- **Solution**: Check workflow permissions in publish jobs

#### 4. Environment protection rules blocking deployment
- **Cause**: GitHub environment requires manual approval
- **Solution**: Review and approve the deployment in GitHub UI

### Getting Help

- **PyPI Trusted Publishing Docs**: https://docs.pypi.org/trusted-publishers/
- **GitHub OIDC Docs**: https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect
- **Package Issues**: https://github.com/mandelbro/graphiti-memory/issues

## Security Best Practices

✅ **Use trusted publishing** instead of API tokens
✅ **Separate build from publish** jobs
✅ **Use environment protection** rules
✅ **Review permissions** regularly
✅ **Enable attestation generation**
✅ **Test on TestPyPI first**

## Next Steps

1. **Set up trusted publishers** on PyPI and TestPyPI
2. **Create GitHub environments** with appropriate protection rules
3. **Test the workflow** by pushing to main branch
4. **Create your first release** when ready for production

Your package will then be available for installation with:

```bash
# Direct installation and run
uvx montesmakes.graphiti-memory

# Or install globally
uv tool install montesmakes.graphiti-memory
montesmakes.graphiti-memory --help
```
