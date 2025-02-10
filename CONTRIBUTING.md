# Contributing

Contributions are welcome, and greatly appreciated!
Every little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

[Report bugs](https://gitlab.com/CoSApp/cosapp/issues/new?issuable_template=Bug) on Gitlab.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitLab issues for bugs.
Anything tagged with `bug` and `to do` is open to whoever wants to implement it.

### Implement Features

Look through the GitLab issues for features.
Anything tagged with `enhancement` and `to do` is open to whoever wants to implement it.

### Write Documentation

CoSApp can always use more documentation, whether as part of the official CoSApp docs, in docstrings, or even in blog posts, online articles, and such.

### Submit Feedback

The best way to send feedback is to file a [Feature Proposal](https://gitlab.com/CoSApp/cosapp/issues/new?issuable_template=FeatureProposal).

If you wish to propose a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make its implementation easier.
* Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `cosapp` for local development.

1. Fork the `cosapp` repo on GitLab.

2. Clone your fork locally:
    ```
    git clone https://gitlab.com/your_login/cosapp.git
    ```

3. Install your local copy into a conda environment. Assuming you have conda installed, this is how you set up your fork for local development::
    ```
    conda create -n cosapp python=3.11 scipy pandas jsonschema pytest
    cd cosapp/
    python -m pip install -e .
    ```

4. Create a branch for local development::
    ```
    git checkout -b name-of-your-branch
    ```
    If your revision tackles an existing issue, starting your branch name by the issue number will automatically close the issue after your branch is merged.
    For example, if issue 37 concerns a solver problem, say, you may name your branch `37-solver-fix`, *e.g.*
    Alternatively, you can specify the general category of your revision, followed by its short name, as in `feature/lightsaber`, `bugfix/solver`, `doc/tutorials`, *etc.*

    Now you can make your changes locally.

5. When you are done, check that your revisions pass the tests, by running:
    ```
    pytest
    ```

6. Commit your changes and push your branch to GitLab:
    ```
    git add .
    git commit -m "Brief description of your changes"
    git push origin name-of-your-branch
    ```

7. Submit a pull request through the GitLab website.
    Go to https://gitlab.com/cosapp/cosapp/-/branches, and click the "Merge Request" button next to your branch name.

## Merge Request Guidelines

Before you submit a merge request, please follow these guidelines:

1. The merge request must include tests.
2. If the merge request adds new features, newly added code must come with its own set of tests. The docs should also be updated, if required. Make sure that
   docstrings and type hints are updated.
3. Importantly, write a brief revision letter explaining your changes in the description of the MR.

The CI stack will check that your revision passes the tests for all supported versions of Python, as a necessary condition for merging.

## Tips

To run a subset of tests:
```
pytest cosapp/path/to/dir
```
