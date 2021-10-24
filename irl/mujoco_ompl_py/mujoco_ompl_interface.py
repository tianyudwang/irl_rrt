import sys
import warnings

from enum import Enum
from math import ceil
from typing import Union, Optional

from mujoco_py.cymj import PyMjModel, PyMjData  # noqa
from mujoco_py import MjSim

from irl.mujoco_ompl_py.mujoco_wrapper import getJointInfo, getCtrlRange, getCtrlInfo

try:
    from ompl import base as ob
    from ompl import control as oc

except ImportError:
    from os.path import abspath, dirname, join

    sys.path.insert(
        0, join(dirname(dirname(dirname(abspath(__file__)))), "ompl", "py-bindings")
    )
    from ompl import base as ob
    from ompl import control as oc

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class mjtJoint(Enum):
    """
    _mjtJoint (type)
    https://github.com/openai/mujoco-py/blob/master/mujoco_py/generated/const.py#L85-#L89
    """
    mjJNT_FREE = 0
    mjJNT_BALL = 1
    mjJNT_SLIDE = 2
    mjJNT_HINGE = 3


def make_1D_VecBounds(low: float, high: float, target: Optional[str] = None, tolerance: float = 1e-4) -> ob.RealVectorBounds:
    """
    Create a 1D RealVectorBounds object.
    Apparently OMPL needs bounds
    350 m/s is just a bit supersonic
    50 m/s is pretty fast for most robot parts
    :param low:
    :param high:
    :return: RealVectorBounds
    """
    assert isinstance(low, (int, float))
    assert isinstance(high, (int, float))
    assert low <= high, f"low {low} must be less than or equal high {high}"
    if low == high:
        # Conside this joint are fixed
        # ompl don't allowed RealVectorBounds with low == high
        # add a small value to bound to foreced it to be a fixed state
        if target == "high":
            high += tolerance
        elif target == "low":
            low -= tolerance
        elif target == "both":
            high += tolerance
            low -= tolerance
        elif target is None:
            warnings.warn("\nOMPL don't allowed RealVectorBounds with low == high. Please specify them manually!")
        else:
            raise ValueError(f"target {target} is not valid")
    bounds = ob.RealVectorBounds(1)
    bounds.setLow(low)
    bounds.setHigh(high)
    return bounds


def readOmplStateKinematic(x, si: ob.SpaceInformation, state: ob.CompoundState):
    """

    :param x: vector
    :param si: space information
    :param state:
    """
    css = si.getStateSpace()
    assert css.isCompound()

    # *Make sure the data vector is the right size
    print(f"x: {len(x)}")
    print(f"Compound Space Dim:{css.getDimension()}")
    assert len(x) == css.getDimension()

    xpos = 0

    for i in range(css.getSubspaceCount()):
        subspace = css.getSubspace(i)
        if isinstance(subspace, ob.RealVectorStateSpace):
            n = subspace.getDimension()
            for j in range(n):
                state[i][j] = x[xpos]
            xpos += 1
            break
        elif isinstance(subspace, ob.SO2StateSpace):
            state[i].value = x[xpos]
            xpos += 1
            break
        elif isinstance(subspace, ob.SO3StateSpace):
            raise NotImplementedError("ob.SO3StateSpace")
        elif isinstance(subspace, ob.SE3StateSpace):
            raise NotImplementedError("ob.SE3StateSpace")
        else:
            raise ValueError("Unhandled subspace type.")

    assert xpos == len(x)


def makeCompoundStateSpace(
    m: PyMjModel, include_velocity: bool, lock_space: bool = False,
) -> ob.CompoundStateSpace:
    """
    Create a compound state space from the MuJoCo model.
    Automatically figure out the State Space Type.
    (optionally including velocity)
    :param m: MuJoCo model
    :param include_velocity:
    :return: CoumpoundStateSpace
    """
    # Create the state space (optionally including velocity)
    space = ob.CompoundStateSpace()

    # Iterate over all the joints in the model
    joints = getJointInfo(m)
    vel_spaces = []
    
    # Add a subspace matching the topology of each joint
    next_qpos = 0
    for joint in joints:
        assert joint.range is not None, "Joint range is not specified (weird)"
        bounds = make_1D_VecBounds(low=joint.range[0], high=joint.range[1])
        # Check our assumptions are OK
        if joint.qposadr != next_qpos:
            raise ValueError(
                f"Joint qposadr {joint.qposadr}: Joints are not in order of qposadr."
            )
        next_qpos += 1
        # Crate an appropriate subspace based on the joint type
        # 0: free, 1: ball, 2: slide, 3: hinge
        if joint.type == mjtJoint.mjJNT_FREE.value:
            joint_space = ob.SE3StateSpace()
            vel_spaces.append(ob.RealVectorStateSpace(6))
            next_qpos += 6

        elif joint.type == mjtJoint.mjJNT_BALL.value:
            joint_space = ob.SO3StateSpace()
            raise NotImplementedError("BALL joints are not yet supported!")

        elif joint.type == mjtJoint.mjJNT_HINGE.value:
            if joint.limited:
                # * A hinge with limits is R^1
                joint_space = ob.RealVectorStateSpace(1)
                joint_space.setBounds(bounds)
            else:
                joint_space = ob.SO2StateSpace()
            vel_spaces.append(ob.RealVectorStateSpace(1))

        elif joint.type == mjtJoint.mjJNT_SLIDE.value:
            joint_space = ob.RealVectorStateSpace(1)
            if joint.limited:
                joint_space.setBounds(bounds)
            vel_spaces.append(ob.RealVectorStateSpace(1))
            
        else:
            raise ValueError(f"Unknown joint type {joint.type}")

        # Add the joint subspace to the compound state space
        space.addSubspace(joint_space, 1.0)

    if next_qpos != m.nq:
        raise ValueError(
            f"Total joint dimensions are not equal to nq.\nJoint dims: {next_qpos} vs nq: {m.nq}"
        )

    # Add the joint velocity subspace to the compound state space
    if include_velocity:
        for vel_space in vel_spaces:
            vel_bounds = make_1D_VecBounds(low=-10, high=10)
            vel_space.setBounds(vel_bounds)
            space.addSubspace(vel_space, 1.0)
    # Lock this state space. This means no further spaces can be added as components.
    if lock_space:
        space.lock()
    return space


def makeRealVectorStateSpace(
    m: PyMjModel, include_velocity: bool
) -> ob.RealVectorStateSpace:
    """
    Create a real vector state space from the MuJoCo model.
    (optionally including velocity)
    :param m: MuJoCo model
    :param include_velocity:
    :return: RealVectorStateSpace
    """
    # Create the state space (optionally including velocity)
    joints = getJointInfo(m)

    dim = 2 * len(joints) if include_velocity else len(joints)
    space = ob.RealVectorStateSpace(dim)

    bounds = ob.RealVectorBounds(dim)

    for i, joint in enumerate(joints):
        assert joint.type == mjtJoint.mjJNT_SLIDE.value or (
            joint.type == mjtJoint.mjJNT_HINGE.value and joint.limited
        )
        bounds.setLow(i, joint.range[0])
        bounds.setHigh(i, joint.range[1])

    space.setBounds(bounds)
    return space


def createSpaceInformation(
    m: PyMjModel, include_velocity: bool
) -> Union[ob.SpaceInformation, oc.SpaceInformation, None]:
    """
    Create a space information from the MuJoCo model.
    :param m: MuJoCo model
    :param low: lower bound for control space
    :param high: upper bound for control space
    :return:
    """
    if include_velocity:
        space = makeCompoundStateSpace(m, True)

        # Creat control space
        control_dim = m.nu
        assert control_dim >= 0, "Control dimension should not be negative."
        if control_dim == 0:
            print("No deafult control space. Need to specify manually.")
            return None
        c_space = oc.RealVectorControlSpace(space, control_dim)
        # Set the bounds for the control space
        c_bounds = ob.RealVectorBounds(control_dim)
        c_bounds.setLow(-1.0)
        c_bounds.setHigh(1.0)

        # Handle specific bounds
        for i in range(control_dim):
            r = getCtrlRange(m, i)
            if r.limited:
                c_bounds.setLow(i, r.range[0])
                c_bounds.setHigh(i, r.range[1])
        c_space.setBounds(c_bounds)

        # Combine into Space Information
        si = oc.SpaceInformation(space, c_space)
        si.setPropagationStepSize(m.opt.timestep)
    else:
        space = makeCompoundStateSpace(m, False)
        si = ob.SpaceInformation(space)
    return si


def copyCompoundOmplStateToMujoco(
    state: ob.CompoundState,
    si: ob.SpaceInformation,
    m: PyMjModel,
    d: PyMjData,
    include_velocity: bool,
) -> None:
    """
    Iterate over subspaces of Compound State Space to copy data from state to mjData.
    Copy position state to d->qpos
    Copy velocity state to d->qvel

    :param state:
    :param si:
    :param m:
    :param d:
    :param include_velocity:
    """
    assert si.getStateSpace().isCompound()
    css = si.getStateSpace()
    qpos_i = 0
    qvel_i = 0

    for i in range(css.getSubspaceCount()):
        subspace = css.getSubspace(i)
        substate = state[i]

        subspace_type = subspace.getType()
        if subspace_type == ob.STATE_SPACE_REAL_VECTOR:
            n = subspace.getDimension()

            # Check if the vector does not align on the size of qpos
            # If this happens an assumption has been violated
            if qpos_i < m.nq and qpos_i + n > m.nq:
                raise ValueError(f"RealVectorState does not align on qpos")
            if not include_velocity and qpos_i >= m.nq:
                raise ValueError(
                    f"RealVectorState does not align on qpos(useVelocities = false)"
                )

            # Copy the vector
            for j in range(n):
                # Check if we copy to qpos or qvel
                if qpos_i < m.nq:
                    d.qpos[qpos_i] = substate[j]
                    qpos_i += 1
                else:
                    d.qvel[qvel_i] = substate[j]
                    qvel_i += 1

        elif subspace_type == ob.STATE_SPACE_SO2:
            if qpos_i >= m.nq:
                raise ValueError("SO2 velocity state should not happen.")
            d.qpos[qpos_i] = substate.value
            qpos_i += 1

        elif subspace_type == ob.STATE_SPACE_SO3:
            if qpos_i + 4 > m.nq:
                raise ValueError("SO3 space overflows qpos")
            qpos_i += 4
            raise NotImplementedError("copySO3State not implemented")

        elif subspace_type == ob.STATE_SPACE_SE3:
            if qpos_i + 7 > m.nq:
                raise ValueError("SE3 space overflows qpos")
            qpos_i += 7
            raise NotImplementedError("copySE3State not implemented")

        else:
            raise ValueError(f"Unknown subspace type {subspace_type}")

    if qpos_i != m.nq:
        raise ValueError(f"Size of data {qpos_i} copied did not match m.nq: {m.nq}")

    if include_velocity and (qvel_i != m.nv):
        raise ValueError("Size of data copied did not match m.nv")


def copyRealVectorOmplStateToMujoco(
    state: ob.RealVectorState,
    si: ob.SpaceInformation,
    m: PyMjModel,
    d: PyMjData,
    include_velocity: bool,
) -> None:
    """
    Copy data from state in RealVectorStateSpace to mjData.
    Copy position state to d->qpos
    Copy velocity state to d->qvel
    *Note: This function assumes that all statesis are RealVectorStateSpace

    :param state:
    :param si:
    :param m:
    :param d:
    :param include_velocity:
    """
    if include_velocity:
        for i in range(si.getDimension()):
            if i < si.getStateDimension() * 0.5 - 1:
                d.qpos[i] = state[i]
            else:
                d.qvel[i - m.nq] = state[i]
    else:
        for i in range(si.getDimension()):
            d.qpos[i] = state[i]


def copyOmplStateToMujoco(
    state: ob.State,
    si: ob.SpaceInformation,
    m: PyMjModel,
    d: PyMjData,
    include_velocity: bool,
) -> None:
    """
    Copy data from state in OMPL to mjData.

    :param state:
    :param si:
    :param m:
    :param d:
    :param include_velocity:
    """
    if si.getStateSpace().isCompound():
        copyCompoundOmplStateToMujoco(state, si, m, d, include_velocity)
    elif si.getStateSpace().getType() == ob.STATE_SPACE_REAL_VECTOR:
        copyRealVectorOmplStateToMujoco(state, si, m, d, include_velocity)
    else:
        raise NotImplementedError(
            "Only support `CompoundStateSpace` and `RealVectorStateSpace` for now"
        )


def copyMujocoStateToOmpl(
    m: PyMjModel,
    d: PyMjData,
    si: ob.SpaceInformation,
    state: ob.CompoundState,
    include_velocity: bool,
) -> None:
    """
    Copy data from mjData to OMPL State.

    :param d:
    :param si:
    :param m:
    :param include_velocity:
    """
    assert si.getStateSpace().isCompound()
    css = si.getStateSpace()
    qpos_i = 0
    qvel_i = 0

    # Iterate over subspaces and copy data from mjData to CompoundState
    for i in range(css.getSubspaceCount()):
        subspace = css.getSubspace(i)

        subspace_type = subspace.getType()
        if subspace_type == ob.STATE_SPACE_REAL_VECTOR:
            n = subspace.getDimension()

            # Check if the vector does not align on the size of qpos
            # If this happens an assumption has been violated
            if qpos_i < m.nq and qpos_i + n > m.nq:
                raise ValueError(f"RealVectorState does not align on qpos")
            if not include_velocity and qpos_i >= m.nq:
                raise ValueError(
                    f"RealVectorState does not align on qpos(useVelocities = false)"
                )

            # Copy the vector
            for j in range(n):
                if qpos_i < m.nq:
                    # !This does not work
                    
                    state[j][0] = d.qpos[qpos_i]
                    qpos_i += 1
                else:
                    state[j][0] = d.qvel[qvel_i]
                    qvel_i += 1
        elif subspace_type == ob.STATE_SPACE_SO2:
            if qpos_i >= m.nq:
                raise ValueError("SO2 velocity state should not happen.")
            d.qpos[qpos_i] = state[i].value
            qpos_i += 1

        elif subspace_type == ob.STATE_SPACE_SO3:
            raise NotImplementedError("copySO3State not implemented")

        elif subspace_type == ob.STATE_SPACE_SO3:

            raise NotImplementedError("copySE3State not implemented")

        else:
            raise ValueError(f"Unknown subspace type {subspace_type}")

    if qpos_i != m.nq:
        raise ValueError("Size of data copied did not match m.nq")

    if include_velocity and (qvel_i != m.nv):
        raise ValueError("Size of data copied did not match m.nv")


def copyOmplControlToMujoco(
    control: oc.RealVectorControlSpace.ControlType,
    si: oc.SpaceInformation,
    m: PyMjModel,
    d: PyMjData,
) -> None:
    """
    Copy data from OMPL control to mjData.

    :param control:
    :param si:
    :param m:
    :param d:
    """
    dim = si.getControlSpace().getDimension()

    if dim != m.nu:
        raise ValueError(f"ControlSpace.getDimension() and mjModel.nu do not match in control dim: {dim}, {m.nu}")

    for i in range(dim):
        d.ctrl[i] = control[i]


def copySO3State2Data(
    state,  # ob.SO3StateSpace,
    data,
) -> None:
    # TODO: Check if this is correct
    data[0] = state.w
    data[1] = state.x
    data[2] = state.y
    data[3] = state.z


def copyData2SO3State(
    data,
    state,  # ob.SO3StateSpace,
) -> None:
    # TODO: Check if this is correct
    state.w = data[0]
    state.x = data[1]
    state.y = data[2]
    state.z = data[3]


def copySE3State2Data(
    state,  # ob.SE3StateSpace.StateType,
    data,
) -> None:
    # TODO: Check if this is correct
    data[0] = state.getX()
    data[1] = state.getY()
    data[2] = state.getZ()
    copySO3State2Data(state.rotation(), data + 3)


def copyDate2SE3State(
    data,
    state,  # ob.SE3StateSpace.StateType
) -> None:
    # TODO: Check if this is correct
    state.setX(data[0])
    state.setY(data[1])
    state.setZ(data[2])
    copyData2SO3State(data + 3, state.rotation())


class MujocoStatePropagator(oc.StatePropagator):
    def __init__(
        self,
        si: oc.SpaceInformation,
        sim: MjSim,
        include_velocity: bool,
    ):
        super().__init__(si)
        self.si = si
        self.sim = sim
        self.include_velocity = include_velocity
        self.max_timestep: float = self.sim.model.opt.timestep

    def getSpaceInformation(self) -> oc.SpaceInformation:
        return self.si

    def propagate(
        self, state: ob.State, control: oc.Control, duration: float, result: ob.State
    ) -> None:
        # Copy ompl state to mujoco
        copyOmplStateToMujoco(
            state, self.si, self.sim.model, self.sim.data, self.include_velocity
        )
        # Copy control
        copyOmplControlToMujoco(control, self.si, self.sim.model, self.sim.data)

        # mj->sim_duration(duration)
        self.sim_duration(duration)

        copyMujocoStateToOmpl(self.sim.model, self.sim.data, self.si, result, self.include_velocity)

    def sim_duration(self, duration: float) -> None:
        steps: int = ceil(duration / self.max_timestep)
        self.sim.model.opt.timestep = duration / steps
        for _ in range(steps):
            self.sim.step()

    def canPropagateBackward(self) -> bool:
        return False

    def canSteer(self) -> bool:
        return False


class MujocoStateValidityChecker(ob.StateValidityChecker):
    def __init__(
        self,
        si: oc.SpaceInformation,
        sim: MjSim,
        include_velocity: bool,
    ):
        super().__init__(si)
        self.si = si
        self.sim = sim
        self.include_velocity = include_velocity

    def isValid(self, state: ob.State) -> bool:
        """State Validation Check"""
        #! mj_step or mj_forwardPosition not Found!!
        # maybe should try get state and set state
        copyOmplStateToMujoco(
            state, self.si, self.sim.model, self.sim.data, self.include_velocity
        )
        # self.sim.forward()
        self.sim.step()
        valid = (self.sim.data.ncon == 0)  #  'No contacts should be detected here'
        return valid