from copy import deepcopy
from math import ceil
from typing import Union

from mujoco_py.cymj import PyMjModel, PyMjData
from mujoco_py import MjSim

from ompl import base as ob
from ompl import control as oc

from irl.mujoco_ompl_py.mujoco_wrapper import getJointInfo, getCtrlRange


# _mjtJoint (type)
mjJNT_FREE = 0
mjJNT_BALL = 1
mjJNT_HINGE = 2
mjJNT_SLIDE = 3


def make_1D_VecBounds(low: float = -50, high: float = 50) -> ob.RealVectorBounds:
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
    m: PyMjModel, include_velocity: bool, verbose: bool = False
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

    # Add a subspace matching the topology of each joint
    next_qpos = 0
    for i, joint in enumerate(joints):
        if verbose:
            print(f"Reach Joint: {i}", end="\r")
        # ? what if range is not specified
        bounds = make_1D_VecBounds(low=joint.range[0], high=joint.range[1])

        # Check our assumptions are OK
        if joint.qposadr != next_qpos:
            raise ValueError(
                f"Joint qposadr {joint.qposadr}: Joints are not in order of qposadr."
            )
        next_qpos += 1

        # Crate an appropriate subspace based on the joint type
        # 0: free, 1: ball, 2: hinge, 3: slide
        if joint.type == mjJNT_FREE:
            joint_space = ob.SE3StateSpace()
            vel_spaces = ob.RealVectorStateSpace(6)
            next_qpos += 6

        elif joint.type == mjJNT_BALL:
            joint_space = ob.SO3StateSpace()
            if joint.limited:
                raise NotImplementedError(
                    "OMPL bounds on SO3 spaces are not implemented!"
                )

            # //vel_spaces = ob.RealVectorStateSpace(3)
            # //next_qpos += 3
            raise NotImplementedError("BALL joints are not yet supported!")

        elif joint.type == mjJNT_HINGE:
            if joint.limited:
                # * A hinge with limits is R^1
                joint_space = ob.RealVectorStateSpace(1)
                joint_space.setBounds(bounds)
            else:
                joint_space = ob.SO2StateSpace()
            vel_spaces = ob.RealVectorStateSpace(1)

        elif joint.type == mjJNT_SLIDE:
            joint_space = ob.RealVectorStateSpace(1)
            if joint.limited:
                joint_space.setBounds(bounds)
            vel_spaces = ob.RealVectorStateSpace(1)
        else:
            raise ValueError(f"Unknown joint type {joint.type}")

        # Add the joint subspace to the compound state space
        space.addSubspace(joint_space, 1.0)

        # Add the joint velocity subspace to the compound state space
        if include_velocity:
            vel_bounds = make_1D_VecBounds()
            vel_spaces.setBounds(vel_bounds)
            space.addSubspace(vel_spaces, 1.0)
    # Lock this state space.
    # This means no further spaces can be added as components.
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
        assert joint.type == mjJNT_SLIDE or (
            joint.type == mjJNT_HINGE and joint.limited
        )
        bounds.setLow(i, joint.range[0])
        bounds.setHigh(i, joint.range[1])

    space.setBounds(bounds)
    return space


def createSpaceInformation(
    m: PyMjModel, include_velocity: bool, verbose: bool = False
) -> Union[ob.SpaceInformation, oc.SpaceInformation]:
    """
    Create a space information from the MuJoCo model.
    :param m: MuJoCo model
    :param low: lower bound for control space
    :param high: upper bound for control space
    :return:
    """
    if include_velocity:
        space = makeCompoundStateSpace(m, True, verbose)

        # Creat control space
        control_dim = m.nu
        assert control_dim >= 0, "Control dimension should not be negative."
        if control_dim == 0:
            raise ValueError("No deafult control space. Need to specify manually.")

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
            raise ValueError("Size of data copied did not match m.nq")

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
                    state[j] = d.qpos[qpos_i]
                    qpos_i += 1
                else:
                    state[j] = d.qvel[qvel_i]
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
        raise ValueError("SpaceInformation and mjModel do not match in control dim")

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
        self.si = si
        self.sim = sim
        self.include_velocity = include_velocity
        self.max_timestep: float = self.sim.model.opt.timestep

    def getSpaceInformation(self) -> oc.SpaceInformation:
        return self.si

    def propagate(
        self, state: ob.State, control: oc.Control, duration: float, result: ob.State
    ) -> None:
        # Copy state
        copyOmplStateToMujoco(
            state, self.si, self.sim.model, self.sim.data, self.include_velocity
        )
        # Copy control
        copyOmplControlToMujoco(control, self.si, self.sim.model, self.sim.data)

        self.sim_duration(duration)

        # mj->sim_duration(duration)
        copyMujocoStateToOmpl(self.sim.model, self.sim.data, self.si_, result)

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
        self.si = si
        self.sim = sim
        self.include_velocity = include_velocity

    def isValid(self, state: ob.State) -> bool:
        """State Validation Check"""
        temp_sim = deepcopy(self.sim)
        copyOmplStateToMujoco(
            state, self.si, temp_sim.model, temp_sim.data, self.include_velocity
        )
        temp_sim.step()
        return temp_sim.data.ncon == 0  #  'No contacts should be detected here'