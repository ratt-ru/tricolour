- [ ] Tests added / passed
- [ ] Fully documented, including `HISTORY.rst` for all changes
      and one of the `docs/*-api.rst` files for new API


<details> <summary>
Howto run test cases and lint the code base </summary>
  ```bash
  $ py.test -v -s tricolour
  ```
If the pep8 tests fail, the quickest way to correct
this is to run `autopep8` and then `flake8` and
`pycodestyle` to fix the remaining issues.

```bash
$ pip install -U autopep8 flake8 pycodestyle
$ autopep8 -r -i tricolour
$ flake8 tricolour
$ pycodestyle tricolour
```

</details>


<details>
<summary> Howto build the documentation </summary>

  To build the docs locally:

  ```
  pip install -r requirements.readthedocs.txt
  cd docs
  READTHEDOCS=True make html
  ```
</details>
