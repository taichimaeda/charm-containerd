import importlib.util
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "upgrade_actions",
    Path("src/actions/upgrade-actions.py"),
)
upgrade_actions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upgrade_actions)


@mock.patch.object(upgrade_actions, "_gpu_packages", return_value=[])
@mock.patch.object(upgrade_actions, "remove_state")
@mock.patch.object(upgrade_actions, "reinstall_containerd")
@mock.patch.object(upgrade_actions, "candidate_containerd_version", return_value=(2, 4, 1))
@mock.patch.object(
    upgrade_actions,
    "_dry_run",
    return_value={"containerd.upgrade-available": True},
)
def test_upgrade_containerd(
    dry_run,
    candidate_containerd_version,
    reinstall_containerd,
    remove_state,
    gpu_packages,
):
    """Verify a containerd upgrade uses the shared apt transition."""
    result = upgrade_actions._upgrade(containerd=True, gpu=False)

    candidate_containerd_version.assert_called_once_with()
    reinstall_containerd.assert_called_once_with((2, 4, 1))
    remove_state.assert_called_once_with("containerd.version-published")
    assert result["containerd.upgrade-complete"] is True


@mock.patch.object(upgrade_actions, "service_restart")
@mock.patch.object(upgrade_actions, "install_nvidia_drivers")
@mock.patch.object(upgrade_actions, "is_state", return_value=True)
@mock.patch.object(upgrade_actions, "action_get", return_value={"force": True})
@mock.patch.object(upgrade_actions, "_gpu_packages", return_value=["cuda-drivers"])
@mock.patch.object(
    upgrade_actions,
    "_dry_run",
    return_value={
        "containerd.upgrade-available": False,
        "cuda-drivers.upgrade-available": False,
    },
)
def test_upgrade_gpu_force(
    dry_run,
    gpu_packages,
    action_get,
    is_state,
    install_nvidia_drivers,
    service_restart,
):
    """Verify a forced GPU upgrade restarts containerd."""
    result = upgrade_actions._upgrade(containerd=False, gpu=True)

    install_nvidia_drivers.assert_called_once_with(reconfigure=False)
    service_restart.assert_called_once_with("containerd")
    assert result["cuda-drivers.upgrade-complete"] is True
