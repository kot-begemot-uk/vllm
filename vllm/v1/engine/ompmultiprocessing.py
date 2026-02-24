'''OMP Aware Multiprocessing manager for running multiprocessing.Process()
Copyright (c) 2026 Red Hat Inc
Copyright (c) 2026 Cambridge Greys Ltd
'''

import subprocess
import os
import json

def parse_mask(mask):
    '''Expand a X-Y,Z list'''
    result = []
    for token in mask.split(","):
        try:
            start, finish = token.split("-")
            if start > finish:
                raise IndexError("Invalid Indexes for cpu ranges")
            for cpu in range(int(start), int(finish) + 1):
                result.append(cpu)
        except ValueError:
            result.append(token)
    return set(result)



def enumerate_resources(mask=None):
    '''Enumerate system resources'''
    allowed = os.sched_getaffinity(0)
    if mask is not None:
        allowed = allowed & mask
    lscpu = {"cpus":{}, "cores":{}, "nodes":{}}
    for cpu in json.loads(subprocess.run(["lscpu", "-Je"], check=True,
                          capture_output=True).stdout)["cpus"]:
        if cpu["cpu"] in allowed:
            lscpu["cpus"][cpu["cpu"]] = [cpu]
            core = int(cpu["core"])
            if lscpu["cores"].get(core, None) is None:
                lscpu["cores"][core] = [cpu]
            else:
                lscpu["cores"][core].append(cpu)
            node = int(cpu["node"])
            if lscpu["nodes"].get(node, None) is None:
                lscpu["nodes"][node] = [cpu]
            else:
                lscpu["nodes"][node].append(cpu)
    return lscpu

def produce_cpu_list(cpus, smt=True):
    '''Produce a CPU list with/without SMT pairs - main cpu list case'''
    mask = []
    for key, value in cpus.items():
        exists = False
        if not smt:
            for cpu in mask:
                if cpu == value[0]["core"]:
                    exists = True
                    break
        if not exists:
            mask.append(key)
    return {"mask":set(mask), "available": True}

def produce_cpu_sublist(scpus, smt=True):
    '''Produce a CPU list with/without SMT pairs - resource leaf case'''
    cpu_list = []
    for value in scpus:
        exists = False
        if not smt:
            for cpu in cpu_list:
                if cpu["core"] == value["core"]:
                    exists = True
                    break
        if not exists:
            cpu_list.append(value)
    mask = []
    for cpu in cpu_list:
        mask.append(cpu["cpu"])

    return {"mask":set(mask), "available": True}

def create_omp_places(resources, strategy, smt=True):
    '''Parse CPU topology and generate possible CPU masks'''
    omp_places = []
    if strategy == "all":
        omp_places.append(produce_cpu_list(resources["cpus"], smt))
    elif strategy == "cores":
        for value in resources["cores"].values():
            omp_places.append(produce_cpu_sublist(value, smt))
    elif strategy == "nodes":
        for value in resources["nodes"].values():
            omp_places.append(produce_cpu_sublist(value, smt))
    else:
        raise NotImplementedError("Unknown strategy")

    return omp_places


# pylint: disable=too-few-public-methods
class OMPProcessManager():
    '''OMP aware wrapper to run mp Process()'''
    omp_places = []
    def __init__(self, strategy="nodes", smt=False):
        self.strategy = strategy
        self.smt = smt
        omp_places = []
        vllm_mask = os.environ.get("VLLM_CPU_OMP_THREADS_BIND", None)
        self.setup_omp = vllm_mask != "nobind"
        if self.setup_omp and len(OMPProcessManager.omp_places) == 0:
            if vllm_mask is not None:
                masks = []
                for spec in vllm_mask.split("|"):
                    masks.append(parse_mask(spec))
            else:
                masks = [None]
            for mask in masks:
                resources = enumerate_resources(mask)
                omp_places.extend(
                    create_omp_places(resources, strategy, smt))
            OMPProcessManager.omp_places = sorted(
                omp_places, key=lambda p: len(p["mask"]), reverse=True)

    def run(self, what, *args, **kwargs):
        '''Run arg with correct OMP environment'''
        if self.setup_omp:
            for place in OMPProcessManager.omp_places:
                if place["available"]:
                    place["available"] = False
                    # pylint: disable=consider-using-f-string
                    os.environ["OMP_PLACES"] = "{}".format(place["mask"])
                    if os.environ.get("OMP_NUM_THREADS", None) is None:
                        # pylint: disable=consider-using-f-string
                        os.environ["OMP_NUM_THREADS"] = "{}".format(len(place["mask"]))
                    os.environ["OMP_PROC_BIND"] = "TRUE"
                    return what(*args, **kwargs)
            raise IndexError("Out of OMP places")
        return what(*args, **kwargs)
