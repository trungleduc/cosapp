<!-- You can initiate a merge request before all changes are completed by specifying
the keyword `WIP:` at the beginning of a title. Don't hesitate to do so if you
want to start discussion about your modifications. -->

# Before merging

Before merging, please check the following items are fullfilled:

- [ ] All unit tests pass
- [ ] All modified and created methods/attributes are typed using 
[typing](https://docs.python.org/3/library/typing.html).
- [ ] All methods input parameters and return parameters are described in the 
method docstring using [numpydoc convention](http://numpydoc.readthedocs.io/en/latest/index.html).
- [ ] There are no unused variables (including `inwards` or `outwards`).
- [ ] For each method, there is at least one unit test checking the expected 
default behaviour.
