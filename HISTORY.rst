=======
History
=======

0.1.9 (2022-09-29)
--------------------
* Fix failing test case (need to switch to gdrive)
* Migrate off Google Drive downloads for CI server
* Pin numpy to remove deprecation errors (and possible numerical issues introduced in 1.20+)
* Pin pytest flake to < 5.0.0 to work around tholo/pytest-flake8#87

0.1.8 (2021-02-24)
------------------

* Pin dask-ms and dask versions (:pr:`82`)
* Fix DATA_DESC_ID indexing (:pr:`79`)
* Fix infinite recursion when computing dask metadata (:pr:`79`)
* Removed .travis and .travis.yml (:pr:`71`) and (:pr:`73`)
* Github CI Actions (:pr:`71`)

0.1.7 (2020-01-06)
------------------

* Change license to BSD3 (:pr:`64`)

0.1.4 (2019-07-23)
------------------

* Upgrade to dask-ms (:pr:`59`)

0.1.3 (2019-07-22)
------------------

* Use pytest-flake8 when running test cases (:pr:`56`)
* Fix polarisation flagging for uncalibrated data (:pr:`55`)
* Add ability to flag on total power (:pr:`55`)
* Baseline statistics (:pr:`55`)
* 4K UHF mask (:pr:`55`)
* Add Pull Request Template (:pr:`53`)


0.1.2 (2019-07-08)
------------------

* Boringify logs
* Remove transposing
* enlarge chunk sizes
* Switch to python 3 only. Python 2.7 no longer supported

0.1.1 (2019-06-27)
------------------

* Minor fixes to scan selection and stats computation

0.1.0 (2018-06-01)
------------------

* Optimise memory usage (:pr:`9`)
* Support ignoring initial flags (:pr:`9`)
* Support flagging on Polarised Intensity (:pr:`9`)
* Added YAML configuration (:pr:`4`)
* Added a progress bar (:pr:`3`)
