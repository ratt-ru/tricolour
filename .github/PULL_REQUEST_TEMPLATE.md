- [ ] Tests added / passed
- [ ] Fully documented, including `HISTORY.rst` for all changes
      and one of the `docs/*-api.rst` files for new API


<details>
<summary> Howto run test cases and lint the code base </summary>

```bash
$ py.test --flake8 -v -s tricolour
```

If you encounter flake8 failures, a quick way to correct
this is to run `autopep8` and `flake8` again.

```bash
$ pip install -U autopep8 black
$ autopep8 -r -i tricolour
$ flake8 tricolour
```

</details>


<details>
<summary> Howto build the documentation </summary>

To build the docs locally:

```bash
$ pip install -r requirements.readthedocs.txt
$ cd docs
$ READTHEDOCS=True make html
```

</details>
