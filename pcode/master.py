# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

from pcode.masters.master_fedga import MasterFedGA

def Master(conf):
    return MasterFedGA(conf)
