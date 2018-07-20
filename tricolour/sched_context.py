from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
import logging
from pprint import pprint

import dask

log = logging.getLogger(__name__)


@contextmanager
def scheduler_context(scheduler):
    """ Set the scheduler to use """
    sched_info = {}

    try:
        if scheduler in ("mt", "threads"):
            log.info("Using multithreaded scheduler")
            dask.config.set(scheduler="threads")
            sched_info = {"type": "threaded"}
        elif scheduler in ("mp", "processes"):
            log.info("Using multiprocessing scheduler")
            dask.config.set(scheduler="processes")
            sched_info = {"type": "multiprocessing"}
        else:
            import distributed
            local_cluster = None

            if scheduler == "local":
                local_cluster = distributed.LocalCluster(processes=False)
                address = local_cluster.scheduler_address
            elif scheduler.startswith('tcp'):
                address = scheduler
            else:
                import json

                with open(scheduler, 'r') as f:
                    address = json.load(f)['address']

            log.info("Using distributed scheduler "
                     "with address '{}'".format(address))
            client = distributed.Client(address)
            dask.config.set(scheduler="distributed")
            client.restart()
            sched_info = {
                "type": "distributed",
                "client": client,
                "local_cluster": local_cluster}

        yield
    except Exception:
        log.exception("Error setting up scheduler", exc_info=True)

    finally:
        try:
            sched_type = sched_info["type"]
        except KeyError:
            pass
        else:
            if sched_type == "distributed":
                try:
                    client = sched_info["client"]
                except KeyError:
                    pass
                else:
                    client.close()

                try:
                    local_cluster = sched_info["local_cluster"]
                except KeyError:
                    local_cluster = None

                if local_cluster:
                    local_cluster.close()
