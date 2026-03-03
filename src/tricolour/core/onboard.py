def onboard():
    """Print setup instructions for CI/CD, PyPI publishing, and GitHub configuration."""
    print(
        """
================================================================================
  tricolour — Setup Instructions
================================================================================

Follow these steps to complete the CI/CD and publishing setup for your project.

────────────────────────────────────────────────────────────────────────────────
  Step 1: Create a GitHub Repository
────────────────────────────────────────────────────────────────────────────────

  First, install the GitHub CLI (gh) if you haven't already:

    # macOS
    brew install gh

    # Ubuntu/Debian
    sudo apt install gh

    # Other platforms: https://github.com/cli/cli#installation

  Then authenticate:

    gh auth login

  Push your project to GitHub:

    gh repo create sjperkins/tricolour --public --source=. --push

  Or create the repo manually at https://github.com/new and push:

    git remote add origin git@github.com:sjperkins/tricolour.git
    git push -u origin main

────────────────────────────────────────────────────────────────────────────────
  Step 2: Set Up Trusted Publishing on PyPI
────────────────────────────────────────────────────────────────────────────────

  Use PyPI's "pending trusted publisher" feature to pre-authorize GitHub
  Actions to publish your package. The PyPI project will be created
  automatically on the first successful publish.

  Go to: https://pypi.org/manage/account/publishing/

  Scroll down to "Add a new pending publisher" and fill in:
    PyPI project name: tricolour
    Owner:             sjperkins
    Repository:        tricolour
    Workflow:          publish.yml
    Environment:       pypi

────────────────────────────────────────────────────────────────────────────────
  Step 3: Create GitHub Environment
────────────────────────────────────────────────────────────────────────────────

  The publish workflow requires a GitHub environment named "pypi".

  Go to: https://github.com/sjperkins/tricolour/settings/environments

  Click "New environment" and name it: pypi

────────────────────────────────────────────────────────────────────────────────
  Step 4: Create a GitHub App (for Automated Cab Updates)
────────────────────────────────────────────────────────────────────────────────

  The update-cabs workflow needs to push commits to the repository. A GitHub
  App is required so these commits can bypass branch protection rules.

  a) Create the app:

     Go to: https://github.com/settings/apps → "New GitHub App"

     Settings:
       Name:        tricolour-bot (or any name you like)
       Homepage:    https://github.com/sjperkins/tricolour
       Webhook:     Uncheck "Active" (not needed)
       Permissions: Repository → Contents → Read & write
       Where:       Only on this account

  b) Generate a private key:

     On the app page → "Generate a private key"
     Save the downloaded .pem file.

  c) Install the app on your repository:

     On the app page → "Install App" → Select your repository.

  d) Add secrets to your repository:

     Go to: https://github.com/sjperkins/tricolour/settings/secrets/actions

     Add two secrets:
       APP_ID          → The App ID shown on the app's settings page
       APP_PRIVATE_KEY → The contents of the .pem file you downloaded

────────────────────────────────────────────────────────────────────────────────
  Step 5: Set Up Branch Protection
────────────────────────────────────────────────────────────────────────────────

  Protect your default branch so all changes go through CI.

  Go to: https://github.com/sjperkins/tricolour/settings/rules

  Click "New ruleset" and configure:

    Name:              Protect main
    Enforcement:       Active
    Target:            Default branch
    Rules to enable:
      - Require a pull request before merging
      - Require status checks to pass (add: "Code Quality", "Tests")
      - Block force pushes

    Bypass list:
      - Add your GitHub App (created in Step 4) so it can push
        automated cab updates

  Make sure to click "Create" to save the ruleset.

────────────────────────────────────────────────────────────────────────────────
  Step 6: Make Your First Release
────────────────────────────────────────────────────────────────────────────────

  When you're ready to publish to PyPI:

    uv run tbump 0.0.0

  This will:
    1. Update version in pyproject.toml and __init__.py
    2. Regenerate cab definitions with the release version
    3. Commit, tag, and push
    4. GitHub Actions will build and publish to PyPI and ghcr.io

================================================================================
  That's it! Your CI/CD pipeline is fully configured.
================================================================================

NOTE: Once you've completed the steps above, you can safely delete the
onboard command (cli/onboard.py and core/onboard.py) and remove it from
cli/__init__.py.

Add your own commands following the same pattern — define a CLI function
with type hints and a @stimela_cab decorator, and Stimela cab definitions
will be auto-generated from your CLI definitions via pre-commit hooks.

For more details, see: https://github.com/landmanbester/hip-cargo#readme
"""
    )
