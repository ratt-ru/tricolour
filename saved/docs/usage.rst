=====
Usage
=====

Command Line
------------

The flagger can be run from the command line::

    $ tricolour --help
      usage: tricolour [-h] [-c CONFIG] [-if] [-fs {standard,polarisation}]
                       [-rc ROW_CHUNKS] [-nw NWORKERS]
                       ms

      positional arguments:
        ms                    Measurement Set

      optional arguments:
        -h, --help            show this help message and exit
        -c CONFIG, --config CONFIG
                              YAML config file containing parameters for the flagger
                              in the 'sum_threshold' key. (default: )
        -if, --ignore-flags
        -fs {standard,polarisation}, --flagging-strategy {standard,polarisation}
                              Flagging Strategy. If 'standard' all correlations in
                              the visibility are flagged independently. If
                              'polarisation' the polarised intensity sqrt(Q^2 + U^2
                              + V^2) is calculated and used to flag all correlations
                              in the visibility. (default: standard)
        -rc ROW_CHUNKS, --row-chunks ROW_CHUNKS
                              Hint indicating the number of Measurement Set rows to
                              read in a single chunk. Smaller and larger numbers
                              will tend to respectively decrease or increase both
                              memory usage and computational efficiency (default:
                              10000)
        -nw NWORKERS, --nworkers NWORKERS
                              Number of workers (threads) to use. By default, set to
                              twice the number of logical CPUs on the system. Many
                              workers can also affect memory usage on systems with
                              many cores. (default: 16)

Sample Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A sample configuration file is shown in ``conf/default.yml``:

.. literalinclude:: ../conf/default.yaml
    :language: yaml

Within a Project
----------------

To use Tricolour in a project::

    import tricolour
