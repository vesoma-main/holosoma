import sys

import numpy as np
from loguru import logger

from holosoma.bridge.base.basic_sdk2py_bridge import BasicSdk2Bridge


def _is_hg_robot(robot_type: str) -> bool:
    """HG message format is used by humanoid robots with 35 motors."""
    return any(tag in robot_type for tag in ("g1", "h1-2"))


class UnitreeSdk2Bridge(BasicSdk2Bridge):
    """Unitree SDK bridge implementation.

    Uses unitree_interface (C++ pybind11 binding) when available, falling back
    to unitree_sdk2py (pure Python SDK) on platforms where the binding is not
    compiled (e.g. macOS).
    """

    SUPPORTED_ROBOT_TYPES = {"g1_29dof", "h1", "h1-2", "go2_12dof"}

    def _init_sdk_components(self):
        """Initialize Unitree SDK-specific components."""
        robot_type = self.robot.asset.robot_type

        if robot_type not in self.SUPPORTED_ROBOT_TYPES:
            raise ValueError(f"Invalid robot type '{robot_type}'. Unitree SDK supports: {self.SUPPORTED_ROBOT_TYPES}")

        try:
            self._init_with_binding(robot_type)
            self._use_sdk2py = False
        except ImportError:
            logger.warning("unitree_interface C++ binding not available, falling back to unitree_sdk2py")
            self._init_with_sdk2py(robot_type)
            self._use_sdk2py = True

    # ------------------------------------------------------------------
    # Initialization: C++ binding path
    # ------------------------------------------------------------------

    def _init_with_binding(self, robot_type: str):
        from unitree_interface import (
            LowState,
            MessageType,
            MotorCommand,
            RobotType,
            UnitreeInterface,
            WirelessController,
        )

        robot_type_map = {
            "g1_29dof": RobotType.G1,
            "h1": RobotType.H1,
            "h1-2": RobotType.H1_2,
            "go2_12dof": RobotType.GO2,
        }
        message_type_map = {
            "g1_29dof": MessageType.HG,
            "h1": MessageType.GO2,
            "h1-2": MessageType.HG,
            "go2_12dof": MessageType.GO2,
        }

        interface_name = self.bridge_config.interface or "eth0"
        self.interface = UnitreeInterface(interface_name, robot_type_map[robot_type], message_type_map[robot_type])
        self.low_state = LowState(self.num_motor)
        self.low_cmd = MotorCommand(self.num_motor)
        self.wireless_controller = WirelessController()

    # ------------------------------------------------------------------
    # Initialization: pure-Python SDK path
    # ------------------------------------------------------------------

    def _init_with_sdk2py(self, robot_type: str):
        from unitree_sdk2py.core.channel import ChannelFactory, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.utils.crc import CRC

        if _is_hg_robot(robot_type):
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdType
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateType

            self._sdk2py_low_state = unitree_hg_msg_dds__LowState_()
        else:
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdType
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateType

            self._sdk2py_low_state = unitree_go_msg_dds__LowState_()

        interface_name = self.bridge_config.interface or "eth0"
        if sys.platform == "darwin" and interface_name == "lo":
            interface_name = "lo0"

        domain_id = self.bridge_config.domain_id
        ChannelFactory().Init(domain_id, interface_name)

        self._lowstate_publisher = ChannelPublisher("rt/lowstate", LowStateType)
        self._lowstate_publisher.Init()

        self._lowcmd_subscriber = ChannelSubscriber("rt/lowcmd", LowCmdType)
        self._sdk2py_latest_cmd = None
        self._lowcmd_subscriber.Init(self._sdk2py_cmd_callback, 1)

        self._crc = CRC()
        self.low_cmd = None

        logger.info(f"unitree_sdk2py bridge initialized (interface={interface_name}, domain_id={domain_id})")

    def _sdk2py_cmd_callback(self, msg):
        """DDS callback: stores the latest LowCmd_ received from the inference process."""
        self._sdk2py_latest_cmd = msg

    # ------------------------------------------------------------------
    # low_cmd_handler
    # ------------------------------------------------------------------

    def low_cmd_handler(self, msg=None):
        """Handle Unitree low-level command messages."""
        if self._use_sdk2py:
            self.low_cmd = self._sdk2py_latest_cmd
        else:
            self.low_cmd = self.interface.read_incoming_command()

    # ------------------------------------------------------------------
    # publish_low_state
    # ------------------------------------------------------------------

    def publish_low_state(self):
        """Publish Unitree low-level state using simulator-agnostic interface."""
        positions, velocities, accelerations = self._get_dof_states()
        actuator_forces = self._get_actuator_forces()
        quaternion, gyro, acceleration = self._get_base_imu_data()

        if self._use_sdk2py:
            self._publish_low_state_sdk2py(positions, velocities, accelerations, actuator_forces, quaternion, gyro, acceleration)
        else:
            self._publish_low_state_binding(positions, velocities, accelerations, actuator_forces, quaternion, gyro, acceleration)

    def _publish_low_state_binding(self, positions, velocities, accelerations, actuator_forces, quaternion, gyro, acceleration):
        self.low_state.motor.q = positions.tolist()
        self.low_state.motor.dq = velocities.tolist()
        self.low_state.motor.ddq = accelerations.tolist()
        self.low_state.motor.tau_est = actuator_forces.tolist()

        quat_array = quaternion.detach().cpu().numpy()
        self.low_state.imu.quat = [float(quat_array[i]) for i in range(4)]
        self.low_state.imu.omega = gyro.detach().cpu().numpy().tolist()
        self.low_state.imu.accel = acceleration.detach().cpu().numpy().tolist()
        self.low_state.tick = int(self.sim_time * 1e3)
        self.interface.publish_low_state(self.low_state)

    def _publish_low_state_sdk2py(self, positions, velocities, accelerations, actuator_forces, quaternion, gyro, acceleration):
        state = self._sdk2py_low_state

        for i in range(min(self.num_motor, len(state.motor_state))):
            state.motor_state[i].q = float(positions[i])
            state.motor_state[i].dq = float(velocities[i])
            state.motor_state[i].ddq = float(accelerations[i])
            state.motor_state[i].tau_est = float(actuator_forces[i])

        quat_array = quaternion.detach().cpu().numpy()
        state.imu_state.quaternion = [float(quat_array[i]) for i in range(4)]
        state.imu_state.gyroscope = gyro.detach().cpu().numpy().tolist()
        state.imu_state.accelerometer = acceleration.detach().cpu().numpy().tolist()

        state.tick = int(self.sim_time * 1e3)

        state.crc = self._crc.Crc(state)
        self._lowstate_publisher.Write(state)

    # ------------------------------------------------------------------
    # publish_wireless_controller
    # ------------------------------------------------------------------

    def publish_wireless_controller(self):
        """Publish wireless controller data."""
        super().publish_wireless_controller()

        if self._use_sdk2py:
            return

        if self.joystick is not None:
            self.interface.publish_wireless_controller(self.wireless_controller)

    # ------------------------------------------------------------------
    # compute_torques
    # ------------------------------------------------------------------

    def compute_torques(self):
        """Compute torques using Unitree's command structure."""
        if not (hasattr(self, "low_cmd") and self.low_cmd):
            return self.torques

        try:
            if self._use_sdk2py:
                return self._compute_torques_sdk2py()
            else:
                return self._compute_torques_binding()
        except Exception as e:
            logger.error(f"Error computing torques: {e}")
            raise

    def _compute_torques_binding(self):
        return self._compute_pd_torques(
            tau_ff=self.low_cmd.tau_ff,
            kp=self.low_cmd.kp,
            kd=self.low_cmd.kd,
            q_target=self.low_cmd.q_target,
            dq_target=self.low_cmd.dq_target,
        )

    def _compute_torques_sdk2py(self):
        motor_cmds = self.low_cmd.motor_cmd
        n = min(self.num_motor, len(motor_cmds))
        tau_ff = np.array([motor_cmds[i].tau for i in range(n)])
        kp = np.array([motor_cmds[i].kp for i in range(n)])
        kd = np.array([motor_cmds[i].kd for i in range(n)])
        q_target = np.array([motor_cmds[i].q for i in range(n)])
        dq_target = np.array([motor_cmds[i].dq for i in range(n)])
        return self._compute_pd_torques(tau_ff=tau_ff, kp=kp, kd=kd, q_target=q_target, dq_target=dq_target)
