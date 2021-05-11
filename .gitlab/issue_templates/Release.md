# Release

For releasing `cosapp`, please check the following items are fulfilled:

1. Documentation is up-to-date:
  * [ ] [Package list](docs/source/modules.rst)
  * [ ] [History](HISTORY.md)
  * [ ] [Contributor list](AUTHORS.md)
2. Version bumped
  * [ ] [core](cosapp/core/_version.py)
  * [ ] [conda recipe](conda.recipe/meta.yaml)
  * [ ] [History](HISTORY.md)
  * [ ] [Git tag](https://gitlab.com/cosapp/cosapp/tags) (vX.Y.Z)
3. Check publication after manually triggering a pipeline:
  * [ ] cosapp on [PyPi](https://pypi.org)
  * [ ] cosapp on [conda-forge](https://conda-forge.org)
  * [ ] Documentation updated on [readthedocs](https://cosapp.readthedocs.io/docs/index.html)
