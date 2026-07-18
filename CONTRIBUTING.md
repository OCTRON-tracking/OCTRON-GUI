# Contributing

These Contributing guidelines are based on the ones from [movement](https://movement.neuroinformatics.dev/latest/community/contributing.html).


## Contribution workflow

If you want to contribute to `OCTRON`, please follow the "fork and pull request" workflow.

When you create your own copy (or "fork") of a project, it's like making a new workspace that shares code with the original project. You can read more about [forking in the GitHub docs](https://docs.github.com/en/get-started/quickstart/fork-a-repo).

Once you've made your changes in your copy, you can submit them as a pull request, which is a way to propose changes back to the main project.

If you are not familiar with `git`, we recommend reading up on [this guide](https://docs.github.com/en/get-started/using-git/about-git#basic-git-commands).

### Forking the repository

1. Fork the [OCTRON-GUI](https://github.com/OCTRON-tracking/OCTRON-GUI) repository on GitHub, by clicking on the "Fork" icon on the top right.

2. Clone your fork to your local machine and navigate to the repository folder:

    ```sh
    git clone https://github.com/<your-github-username>/OCTRON-GUI.git
    cd OCTRON-GUI
    ```

3. Define a remote named `upstream`, that links to the base `OCTRON-GUI` repository:
   This allows you to pull the latest changes from the main repository.

    ```sh
    git remote add upstream https://github.com/OCTRON-tracking/OCTRON-GUI.git
    ```

    :::{note}
    Your repository now has two remotes: `origin` (the fork under your user account, where you push changes) and `upstream` (the main repository, where you pull updates from)
    :::

### Creating a development environment

Now that you have the repository locally, you need to set up a Python environment and install the project dependencies.

1. Create an environment using [conda](https://docs.conda.io/projects/conda/en/latest/#) and install `OCTRON-GUI` in editable mode, including development dependencies.

    First, create and activate a `conda` environment:

    ```sh
    conda create -n octron-dev -c conda-forge python=3.11
    conda activate octron-dev
    ```

    Then, install the package in editable mode with development dependencies:

    ```sh
    pip install -e ".[dev]"
    ```

2. Finally, initialise the pre-commit hooks:

    ```sh
    pre-commit install
    ```

### Pull requests
In all cases, please submit code to the main repository via a pull request (PR).
We recommend, and adhere, to the following conventions:

- Please submit _draft_ PRs as early as possible to allow for discussion.
- The PR title should be descriptive e.g. "Add new function to do X" or "Fix bug in Y".
- The PR description should be used to provide context and motivation for the changes.
  - If the PR is solving an issue, please add the issue number to the PR description, e.g. "Fixes #123" or "Closes #123".
  - Make sure to include cross-links to other relevant issues, PRs and Zulip threads, for context.
- The maintainers triage PRs and assign suitable reviewers using the GitHub review system.
- One approval of a PR (by a maintainer) is enough for it to be merged.
- Unless someone approves the PR with optional comments, the PR is immediately merged by the approving reviewer.
- PRs are preferably merged via the ["squash and merge"](github-docs:pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges#squash-and-merge-your-commits) option, to keep a clean commit history on the _main_ branch.

A typical PR workflow would be:
* Create a new branch, make your changes, and stage them.
* When you try to commit, the [pre-commit hooks](#formatting-and-pre-commit-hooks) will be triggered.
* Stage any changes made by the hooks, and commit.
* You may also run the pre-commit hooks manually, at any time, with `pre-commit run -a`.
* Make sure to write tests for any new features or bug fixes. See [testing](#testing) below.
* Don't forget to update the documentation, if necessary. See [contributing documentation](#contributing-documentation) below.
* Push your changes to your fork on GitHub(`git push origin <branch-name>`).
* Open a draft pull request from your fork to the upstream `OCTRON-GUI` repository, with a meaningful title and a thorough description of the changes.
  :::{note}
  When creating the PR, ensure the base repository is `OCTRON-tracking/OCTRON-GUI` (the `upstream`) and the head repository is your fork. GitHub sometimes defaults to comparing against your own fork. Also make sure to tick the "Allow edits by maintainers" checkbox, so that maintainers can make small fixes directly to your branch.
  :::
* If all checks (e.g. linting, type checking, testing) run successfully, you may mark the pull request as ready for review.
* Respond to review comments and implement any requested changes. This could be a couple of rounds of feedback.
* When all is good, one of the maintainers will approve the PR and merge it.
* Success 🎉 !! Your PR will be (squash-)merged into the _main_ branch.


## Development guidelines

### Formatting and pre-commit hooks
Running `pre-commit install` will set up [pre-commit hooks](https://pre-commit.com/) to ensure a consistent formatting style. Currently, these include:
* [ruff](https://github.com/astral-sh/ruff) does a number of jobs, including code linting and auto-formatting.
* [mypy](https://mypy.readthedocs.io/en/stable/index.html) as a static type checker.
* [check-manifest](https://github.com/mgedmin/check-manifest) to ensure that the right files are included in the pip package.
* [codespell](https://github.com/codespell-project/codespell) to check for common misspellings.

These will prevent code from being committed if any of these hooks fail.
To run all the hooks before committing:

```sh
pre-commit run  # for staged files
pre-commit run -a  # for all files in the repository
```

Some problems will be automatically fixed by the hooks. In this case, you should
stage the auto-fixed changes and run the hooks again:

```sh
git add .
pre-commit run
```

If a problem cannot be auto-fixed, the corresponding tool will provide
information on what the issue is and how to fix it. For example, `ruff` might
output something like:

```sh
octron/cli.py:551:80: E501 Line too long (90 > 79)
```

This pinpoints the problem to a single code line and a specific [ruff rule](https://docs.astral.sh/ruff/rules/) violation.
Sometimes you may have good reasons to ignore a particular rule for a specific line of code. You can do this by adding an inline comment, e.g. `# noqa: E501`. Replace `E501` with the code of the rule you want to ignore.

### Docstrings
We adhere to the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style.
All public functions, classes, and methods must include docstrings.

To document module‑level variables or class attributes, place a string literal immediately after the definition; see [PEP 257](https://peps.python.org/pep-0257/#what-is-a-docstring):

```python
class MyClass:
    x: int = 42
    """Description of x."""
```

### Testing
We use [pytest](https://docs.pytest.org/en/latest/) for testing. All new features should be accompanied by tests.

Tests are stored in the `tests` directory, structured as follows:

- `test_unit/`: contains unit tests that aim to follow the `OCTRON-GUI` package structure.
- `test_integration/`: includes tests for interactions between different modules.
- `fixtures/`: holds reusable test data fixtures, automatically imported via `conftest.py`. Check for existing fixtures before adding new ones, to avoid duplication.


### Continuous integration
All pushes and pull requests will be built by [GitHub actions](https://docs.github.com/en/actions#).
This will usually include linting, testing and deployment.

A GitHub actions workflow (`.github/workflows/test_and_deploy.yml`) has been set up to run (on each push/PR):
* Linting checks (pre-commit).
* Testing (only if linting checks pass)
<!-- NOT IMPLEMENTED YET: * Release to PyPI (only if a git tag is present and if tests pass). -->

### Versioning and releases
We use [semantic versioning](https://semver.org/), where each version number has the form `MAJOR`.`MINOR`.`PATCH`:

* PATCH = small bugfix
* MINOR = new feature
* MAJOR = breaking change

Rather than editing the version number by hand anywhere in the code, we let
[setuptools_scm](https://setuptools-scm.readthedocs.io/en/latest/#) derive it
automatically from git tags. This is configured under `[tool.setuptools_scm]` in
`pyproject.toml`, and works as follows:

* **On a tagged commit**, the version is the tag itself (e.g. tag `v1.0.0` → version `1.0.0`).
* **On commits after a tag**, the version is derived from the most recent tag
  reachable from the current commit, plus the number of commits since that tag.
  We use the `post-release` scheme with `no-local-version`, so this looks like 
  `1.0.0.post3` ("3 commits after 1.0.0"), without the default trailing `+<commit-hash>`
  segment.

So the version is always determined by the latest tag reachable from the commit
being built.

To make a new release, create and push a tag. Make sure you first commit any
changes you want included. For example, to release version `1.0.0`:

```sh
git add .
git commit -m "Add new changes"
git tag -a v1.0.0 -m "Bump to version 1.0.0"
git push --follow-tags
```

Alternatively, you can create the release and tag through the GitHub web interface.

<!-- NOT IMPLEMENTED YET: Pushing the tag then triggers the package's deployment to PyPI. -->


## Contributing documentation
The documentation is hosted via [GitHub pages](https://pages.github.com/) at
[octron-tracking.github.io/OCTRON-docs/](https://octron-tracking.github.io/OCTRON-docs/). Its source files are located in a separate repository, [OCTRON-docs](https://github.com/OCTRON-tracking/OCTRON-docs). Please refer to that repo's [README](https://github.com/OCTRON-tracking/OCTRON-docs#octron-docs) for guidelines on how to contribute.
