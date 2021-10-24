from typing import List

import numpy as np
from mujoco_py.cymj import PyMjModel


mjJNT = {
    0: "FREE" ,
    1: "BALL" ,
    2: "SLIDE",
    3: "HINGE",
}

class JointInfo:
    """
    A class to store information about a joint in a MuJoCo model.
    """
    __slots__ = ["index", "name", "type", "limited", "range", "qposadr", "dofadr", "mjJNT"] 
    
    def __init__(self):
        self.index: int
        self.name: str
        self.type: int
        self.limited: bool
        self.qposadr: int
        self.dofadr: int
        self.range: List[float, float] = [None, None]

    def __repr__(self):
        return (
            f"Joint {self.index}: (name:{self.name[-4:]}, "
            + f"type:{self.type} ({mjJNT[self.type]}), "
            + f"limit:{self.limited}, "
            + f"range:{self.range}, "
            + f"qposadr:{self.qposadr}, "
            + f"dofadr:{self.dofadr})\n"
        )


class StateRange:
    """
    A class to store information about a state in a MuJoCo model.
    """

    __slots__ = ["index", "range", "limited"]

    def __init__(self):
        self.index: int
        self.limited: bool
        self.range: List[float, float] = [None, None]

    def __repr__(self):
        return f"StateRange {self.index}: (range:{self.range}, " + f"limit:{self.limited},\n"


def getJointInfo(m: PyMjModel):
    """
    Obtain information about the joints in the Mujoco model.
    """
    joints = []
    for i in range(m.njnt):
        joint = JointInfo()
        joint.index = i
        joint.name = "".join(m.names.astype(str)) + " " + str(m.name_jntadr[i])
        joint.type = m.jnt_type[i]
        joint.limited = bool(m.jnt_limited[i])
        joint.range[0] = np.asarray(m.jnt_range).flatten()[2 * i]
        joint.range[1] = np.asarray(m.jnt_range).flatten()[2 * i + 1]
        joint.qposadr = m.jnt_qposadr[i]
        joint.dofadr = m.jnt_dofadr[i]
        joints.append(joint)
    return joints


def getCtrlRange(m: PyMjModel, i: int) -> StateRange:
    """
    Obtain the control range for a given joint in the Mujoco model.
    """
    assert m.actuator_ctrllimited is not None, "actuator's ctrl limited not specified"
    assert m.actuator_ctrlrange is not None, "actuator's ctrl range not specified"
    r = StateRange()
    r.index = i
    r.limited = bool(m.actuator_ctrllimited[i])
    r.range[0] = np.asarray(m.actuator_ctrlrange).flatten()[2 * i]
    r.range[1] = np.asarray(m.actuator_ctrlrange).flatten()[2 * i + 1]
    return r

def getCtrlInfo(m: PyMjModel) -> List[StateRange]:
    """
    Obtain information about the ctrls in the Mujoco model.
    """
    ctrls = []
    for i in range(m.nu):
        ctrls.append(getCtrlRange(m, i))
    return ctrls