import platform
import sys

from setuptools import setup

UNITREE_VERSION = "0.1.2"
UNITREE_REPO = "https://github.com/amazon-far/unitree_sdk2"
BOOSTER_VERSION = "0.1.0"
BOOSTER_REPO = "https://github.com/amazon-far/booster_robotics_sdk"

PLATFORM_MAP = {
    "x86_64": "linux_x86_64",
    "aarch64": "linux_aarch64",
}

pyvers = f"cp{sys.version_info.major}{sys.version_info.minor}"
platform_str = PLATFORM_MAP.get(platform.machine(), "linux_x86_64")

# Build extras_require based on platform
extras_require = {}

# Only include SDKs on Linux (they don't have macOS wheels)
if platform.system() == "Linux":
    unitree_url = f"{UNITREE_REPO}/releases/download/{UNITREE_VERSION}/unitree_sdk2-{UNITREE_VERSION}-{pyvers}-{pyvers}-{platform_str}.whl"  # noqa: E501
    booster_url = f"{BOOSTER_REPO}/releases/download/{BOOSTER_VERSION}/booster_robotics_sdk-{BOOSTER_VERSION}-{pyvers}-{pyvers}-{platform_str}.whl"  # noqa: E501
    extras_require["unitree"] = [f"unitree_sdk2 @ {unitree_url}"]
    extras_require["booster"] = [f"booster_robotics_sdk @ {booster_url}"]
else:
    # macOS: unitree/booster extras are empty (installed separately from source)
    extras_require["unitree"] = []
    extras_require["booster"] = []

# Build entry points - only register bridges on Linux
# (on macOS, unitree_sdk2 bridge will work but installed from source)
entry_points = {"holosoma.bridge": []}
if platform.system() == "Linux":
    entry_points["holosoma.bridge"].extend([
        "unitree = holosoma.bridge.unitree:UnitreeSdk2Bridge",
        "booster = holosoma.bridge.booster:BoosterSdk2Bridge",
    ])
else:
    # On macOS, only unitree bridge (when unitree_sdk2 installed from source)
    entry_points["holosoma.bridge"].append(
        "unitree = holosoma.bridge.unitree:UnitreeSdk2Bridge"
    )

setup(
    extras_require=extras_require,
    entry_points=entry_points,
)
