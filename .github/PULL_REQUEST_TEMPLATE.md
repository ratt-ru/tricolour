- [ ] Tests added / passed
<detail>
<summary> Run test cases </summary>
  ```bash
  $ py.test -v -s tricolour
  ```
</detail>

<detail>
<summary> linting the code base </summary>
If the pep8 tests fail, the quickest way to correct
this is to run `autopep8` and then `flake8` and
`pycodestyle` to fix the remaining issues.

```
$ pip install -U autopep8 flake8 pycodestyle
$ autopep8 -r -i tricolour
$ flake8 tricolour
$ pycodestyle tricolour
```
</detail>

- [ ] Fully documented, including `HISTORY.rst` for all changes
      and one of the `docs/*-api.rst` files for new API

<detail>
<summary> Building the documentation </summary>

  To build the docs locally:

  ```
  pip install -r requirements.readthedocs.txt
  cd docs
  READTHEDOCS=True make html
  ```
</detail>
