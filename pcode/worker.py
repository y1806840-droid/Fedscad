# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

from pcode.workers.worker_fedga import WorkerFedGA

def Worker(conf):
    return WorkerFedGA(conf)
