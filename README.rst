=========
Tricolour
=========

A variant of the Science Data Processing flagging code, wrapped in dask,
operating on Measurement Sets.

Copyright SDP / RARG, 2018-2019. South African Radio Astronomy Observatory (SARAO)

Installation
------------

The following lines should install tricolour (preferably in your virtual environment)

    pip install -e <path to your virtual environment>

This will automatically include a default strategy and static masks.

Usage
------------
The default configuration in will be searched for in the following paths:

    /usr/etc/tricolour
    <tricolour package installation directory>/conf
    ~/.config/tricolour
    ~/.tricolour

This behaviour can be overwritten with the '-c' argument to the flagger.
By default the installation will provide a default configuration in

    <tricolour package installation directory>/conf

The strategy is a yaml dictionary keyed flagger step, e.g.

    reset_flags:
    
        task: unflag
        
        order: 0
        
    background_static_mask:
    
        task: apply_static_mask
        
        order: 1
        
        accumulation_mode: "or"
        
        uvrange: "0~350m"
        
    background_flags:
    
        task: sum_threshold 
        
        order: 2
        
        outlier_nsigma: 4.5
        
        windows_time: [1, 2, 4, 8]
        
        windows_freq: [1, 2, 4, 8]
        
        background_reject: 2.0
        
        background_iterations: 1
        
        spike_width_time: 12.5
        
        spike_width_freq: 10.0
        
        time_extend: 3
        
        freq_extend: 3
        
        freq_chunks: 10
        
        average_freq: 1
        
        flag_all_time_frac: 0.6
        
        flag_all_freq_frac: 0.8
        
        rho: 1.3
        
        num_major_iterations: 10

Each such named flagging step has a corresponding supported task and keyword parameters to be passed to that task. At the moment the the data is divided on the field, data descriptor ('SPW' / 'IF') and scan boundaries.

The following arguments can be specified to the flagger:

    usage: tricolour [-h] [-c CONFIG] [-if] [-fs {standard,polarisation}]
                     [-rc ROW_CHUNKS] [-nw NWORKERS] [-dm DILATE_MASKS]
                     [-dc DATA_COLUMN] [-fn FIELD_NAMES] [-dpm]
                     ms

    positional arguments:
      ms                    Measurement Set

    optional arguments:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            YAML config file containing parameters for the flagger
                            in the 'sum_threshold' key. (default: /home/hugo/OPTSO
                            FT/tricolour/tricolour/conf/default.yaml)
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
                            many cores. (default: 112)
      -dm DILATE_MASKS, --dilate-masks DILATE_MASKS
                            Number of channels to dilate as int or string with
                            units (default: None)
      -dc DATA_COLUMN, --data-column DATA_COLUMN
                            Name of visibility data column to flag (default: DATA)
      -fn FIELD_NAMES, --field-names FIELD_NAMES
                            Name(s) of fields to flag. Defaults to flagging all
                            (default: [])
      -dpm, --disable-post-mortem
                            Disable the default behaviour of starting the
                            Interactive Python Debugger upon an unhandled
                            exception. This may be necessary for batch pipelining
                            (default: False)
