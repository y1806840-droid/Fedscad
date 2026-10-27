# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

from pcode.masters.master_fedscad import MasterFedSCAD

def Master(conf):
    return MasterFedSCAD(conf)
