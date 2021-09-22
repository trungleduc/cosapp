# Release

## Requirement checklist

1. Documentation is up-to-date:
  * [ ] [Package list](docs/source/modules.rst)
  * [ ] [History](HISTORY.md)
  * [ ] [Contributor list](AUTHORS.md)

2. Version bumped
  * [ ] [core](cosapp/core/_version.py)
  * [ ] [conda recipe](conda.recipe/meta.yaml)
  * [ ] [History](HISTORY.md)

## Deployment after merge

The deployment pipeline is automatically triggered by new tags, or can be triggered manually.
  * [ ] [Git tag](https://gitlab.com/cosapp/cosapp/tags) (vX.Y.Z)
  * [ ] cosapp on [PyPi](https://pypi.org)
  * [ ] cosapp on [Anaconda](https://anaconda.org/cosapp/cosapp)
  * [ ] cosapp on [conda-forge](https://conda-forge.org) (Pull Request on [feedstock](https://github.com/conda-forge/cosapp-feedstock))
  * [ ] Documentation updated on [readthedocs](https://cosapp.readthedocs.io/docs/index.html)
