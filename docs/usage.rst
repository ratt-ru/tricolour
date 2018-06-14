=====
Usage
=====

Command Line
------------

The flagger can be run from the command line::

    $ tricolour --help
      usage: tricolour [-h] [-c CONFIG] [-if] [-fs {standard,polarisation}] ms

      positional arguments:
        ms                    Measurement Set

      optional arguments:
        -h, --help            show this help message and exit
        -c CONFIG, --config CONFIG
                              YAML config file containing parameters for the flagger
                              in the 'sum_threshold' key.
        -if, --ignore-flags
        -fs {standard,polarisation}, --flagging-strategy {standard,polarisation}
                              Flagging Strategy. If 'standard' all correlations in
                              the visibility are flagged independently. If
                              'polarisation' the unpolarised intensity is calculated
                              and used to flag all correlations in the visibility.

Sample Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A sample configuration file is shown in ``conf/default.yml``:

.. literalinclude:: ../conf/default.yaml
    :language: yaml

Within a Project
----------------

To use Tricolour in a project::

    import tricolour
