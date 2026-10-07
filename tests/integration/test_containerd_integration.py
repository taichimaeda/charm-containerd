import asyncio
import json
import logging
import uuid

import jinja2
from juju.unit import Unit
from juju.application import Application
from pathlib import Path
import pytest
import pytest_asyncio
import re
import shlex
import toml
from typing import Dict
from pytest_operator.plugin import OpsTest
from tenacity import retry, stop_after_attempt, wait_exponential
from utils import JujuRun
import yaml

log = logging.getLogger(__name__)


def format_kubectl_cmd(cmd: str) -> str:
    """Return a kubectl command with the kubeconfig path."""
    return f"kubectl --kubeconfig /root/.kube/config {cmd}"


@pytest.mark.abort_on_fail
async def test_build_and_deploy(ops_test):
    """Build and deploy Containerd in bundle."""
    log.info("Build Charm...")
    charm = next(Path.cwd().glob("containerd*.charm"), None)
    if not charm:
        log.info("Build Charm...")
        charm = await ops_test.build_charm(".")

    overlays = [
        ops_test.Bundle("kubernetes-core", channel="1.36/edge"),
        Path("tests/data/charm.yaml"),
    ]

    log.info("Build Bundle...")
    bundle, *overlays = await ops_test.async_render_bundles(*overlays, charm=charm)

    log.info("Deploy Bundle...")
    model = ops_test.model_full_name
    cmd = f"juju deploy -m {model} {bundle} "
    cmd += " ".join(f"--overlay={f}" for f in overlays)
    rc, stdout, stderr = await ops_test.run(*shlex.split(cmd))
    assert rc == 0, f"Bundle deploy failed: {(stderr or stdout).strip()}"

    apps = [app for fragment in (bundle, *overlays) for app in yaml.safe_load(fragment.open())["applications"]]
    await ops_test.model.wait_for_idle(apps=apps, status="active", timeout=60 * 60)


async def test_status_messages(ops_test):
    """Validate that the status messages are correct."""
    for unit in ops_test.model.applications["containerd"].units:
        assert unit.workload_status == "active"
        assert unit.workload_status_message == "Container runtime available"


async def process_elapsed_time(unit, process):
    """Get elasped time of a running process."""
    result = await JujuRun.command(unit, f"ps -p `pidof {process}` -o etimes=")
    return int(result.stdout)


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
async def pods_in_state(unit: Unit, selector: Dict[str, str], state: str = "Running"):
    """Retry checking until the pods all match a specified state."""
    format = ",".join("=".join(pairs) for pairs in selector.items())
    cmd = format_kubectl_cmd(f"get pods -l={format} --no-headers")
    result = await JujuRun.command(unit, cmd)
    pod_set = result.stdout.splitlines()
    assert pod_set and all(state in line for line in pod_set)
    return pod_set


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(12),
    reraise=True,
)
async def nodes_in_state(unit: Unit, count: int, state: str):
    """Retry checking until all nodes match the specified state."""
    result = await JujuRun.command(unit, format_kubectl_cmd("get nodes"))
    assert result.stdout.count(state) == count
    return result


@pytest.mark.parametrize("which_action", ("containerd", "packages"))
async def test_upgrade_action(ops_test, which_action):
    """Test running upgrade action."""
    unit = ops_test.model.applications["containerd"].units[0]
    start = await process_elapsed_time(unit, "containerd")
    output = await JujuRun.action(unit, f"upgrade-{which_action}")
    results = output.results
    log.info(f"Upgrade results = '{results}'")
    assert results["containerd"]["available"] == results["containerd"]["installed"]
    assert results["containerd"]["upgrade-available"] == "False"
    assert not results["containerd"].get("upgrade-completed"), "No upgrade should have been run"
    end = await process_elapsed_time(unit, "containerd")
    assert end >= start, "containerd service shouldn't have been restarted"


@pytest.mark.parametrize("which_action", ("containerd", "packages"))
async def test_upgrade_dry_run_action(ops_test, which_action):
    """Test running upgrade action in dry-run mode."""
    unit = ops_test.model.applications["containerd"].units[0]
    start = await process_elapsed_time(unit, "containerd")
    output = await JujuRun.action(unit, f"upgrade-{which_action}", **{"dry-run": True})
    results = output.results
    log.info(f"Upgrade dry-run results = '{results}'")
    assert results["containerd"]["available"] == results["containerd"]["installed"]
    assert results["containerd"]["upgrade-available"] == "False"
    end = await process_elapsed_time(unit, "containerd")
    assert end >= start, "containerd service shouldn't have been restarted"


async def test_upgrade_action_containerd_force(ops_test):
    """Test running upgrade action without GPU and with force."""
    unit = ops_test.model.applications["containerd"].units[0]
    start = await process_elapsed_time(unit, "containerd")
    action = await JujuRun.action(unit, "upgrade-packages", force=True)
    results = action.results
    log.info(f"Upgrade results = '{results}'")
    assert results["containerd"]["available"] == results["containerd"]["installed"]
    assert results["containerd"]["upgrade-available"] == "False"
    assert not results["containerd"].get("upgrade-completed"), "No upgrade should have been run"
    end = await process_elapsed_time(unit, "containerd")
    assert end >= start, "containerd service shouldn't have been restarted"


async def test_upgrade_action_gpu_uninstalled_but_gpu_forced(ops_test):
    """Test running GPU force upgrade-action with no GPU drivers installed.

    upgrade-action with `GPU` and `force` flags both set but without GPU drivers currently
    installed should not upgrade any GPU drivers.
    """
    unit = ops_test.model.applications["containerd"].units[0]
    start = await process_elapsed_time(unit, "containerd")
    action = await JujuRun.action(unit, "upgrade-packages", containerd=False, gpu=True, force=True)
    results = action.results
    log.info(f"Upgrade results = '{results}'")
    action = await JujuRun.command(unit, "dpkg-query --list cuda-drivers", check=False)
    assert "cuda-drivers" in action.stderr, "cuda-drivers shouldn't be installed"
    end = await process_elapsed_time(unit, "containerd")
    assert end >= start, "containerd service shouldn't have been restarted"


@pytest_asyncio.fixture(scope="module")
async def juju_config(ops_test):
    """Apply configuration for a test, then revert after the test is completed."""

    async def setup(application, _timeout=10 * 60, **new_config):
        """Apply config by application name and the config values.

        @param: str application: name of app to configure
        @param: dict new_config: configuration key=values to adjust
        @param: float  _timeout: time in seconds to wait for applications to be stable
        """
        await update_reverts(application, new_config.keys(), _timeout)
        await ops_test.model.applications[application].set_config(new_config)
        await ops_test.model.wait_for_idle(apps=[application], status="active", timeout=_timeout)

    async def update_reverts(application, configs, _timeout):
        """Control what config is reverted per app during the test module teardown.

        Because juju_config is a module scoped fixture, it isn't torn down until all the tests
        in the module are completed. The `setup` method could be called multiple times
        by various tests, but only the first call should gather the original config

        Subsequent calls, should update which keys are reverted, and the greatest timeout
        selected to revert all keys.
        """
        reverts = to_revert.get(application)
        if not reverts:
            reverts = (await ops_test.model.applications[application].get_config(), set(configs), _timeout)
        else:
            reverts = (reverts[0], reverts[1] | set(configs), max(reverts[2], _timeout))
        to_revert[application] = reverts

    to_revert = {}
    yield setup
    for app, (pre_test, settable, timeout) in to_revert.items():
        revert_config = {key: pre_test[key]["value"] for key in settable}
        await ops_test.model.applications[app].set_config(revert_config)
    await ops_test.model.wait_for_idle(apps=list(to_revert.keys()), status="active")


async def containerd_config(unit):
    """Gather containerd config and load as a dict from its toml representation."""
    output = await JujuRun.command(unit, "cat /etc/containerd/config.toml")
    assert output.stdout, "Containerd output was empty"
    return toml.loads(output.stdout)


def containerd_config_version(config):
    """Return the rendered containerd config version."""
    return config.get("version", 1)


async def containerd_version(unit):
    """Return the containerd version."""
    output = await JujuRun.command(unit, "containerd --version")
    match = re.search(r"(?<!\d)v?(\d+\.\d+\.\d+)(?!\d)", output.stdout)
    assert match, f"Unable to parse containerd version: {output.stdout}"
    return match.group(1)


async def containerd_major_version(unit):
    """Return the containerd major version."""
    version = await containerd_version(unit)
    return int(version.split(".", 1)[0])


async def containerd_registry_config(unit, host):
    """Load a registry hosts.toml file."""
    output = await JujuRun.command(unit, f"cat /etc/containerd/certs.d/{host}/hosts.toml")
    return toml.loads(output.stdout)


def registry_host(ops_test, application="docker-registry"):
    """Return the related registry host."""
    registry_unit = ops_test.model.applications[application].units[0]
    # Example: "Ready at 10.22.129.111:5000 (https)."
    message = registry_unit.workload_status_message
    match = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}:\d+\b", message)
    assert match, f"Unable to parse registry host: {message}"
    return match.group()


async def run_image(unit, name, image):
    """Run a pod with an image."""
    delete = format_kubectl_cmd(f"delete pod {name} --ignore-not-found")
    await JujuRun.command(unit, delete)
    try:
        run = format_kubectl_cmd(
            f"run {name} "
            f"--image={shlex.quote(image)} "
            "--image-pull-policy=Always "
            "--restart=Never "
            "--command -- sleep 300"
        )
        await JujuRun.command(unit, run)

        wait = format_kubectl_cmd(f"wait --for=condition=Ready pod/{name} --timeout=5m")
        await JujuRun.command(unit, wait)
    finally:
        await JujuRun.command(unit, delete, check=False)


@pytest.fixture(scope="module")
def public_registry_image():
    """Return an image from a public registry."""
    return "docker.io/library/busybox:1.38.0"


@pytest_asyncio.fixture(scope="module")
async def private_registry_image(ops_test, public_registry_image):
    """Push an image to the test registry."""
    unit = ops_test.model.applications["docker-registry"].units[0]
    action = await JujuRun.action(unit, "push", image=public_registry_image)
    assert action.results["outcome"] == "success"
    return action.results["raw"].removeprefix("pushed ")


@pytest_asyncio.fixture()
async def custom_registry_image(ops_test, juju_config, public_registry_image):
    """Push an image to the custom registry."""
    # Disable authentication while seeding the registry.
    await juju_config("custom-registry", **{"auth-basic-password": ""})

    registry_unit = ops_test.model.applications["custom-registry"].units[0]
    action = await JujuRun.action(registry_unit, "push", image=public_registry_image)
    assert action.results["outcome"] == "success"
    return action.results["raw"].removeprefix("pushed ")


@pytest_asyncio.fixture()
async def custom_registry_config(ops_test):
    """Return TLS config for the custom registry."""
    unit = ops_test.model.applications["containerd"].units[0]
    host = registry_host(ops_test, "custom-registry")
    config = await containerd_config(unit)
    config_version = containerd_config_version(config)

    if config_version == 1:
        configs = config["plugins"]["cri"]["registry"]["configs"]
    if config_version == 2:
        configs = config["plugins"]["io.containerd.grpc.v1.cri"]["registry"]["configs"]
    if config_version in (1, 2):
        related_host = registry_host(ops_test)
        registry = next(value for key, value in configs.items() if related_host in key)
        ca = registry["tls"]["ca_file"]
        cert = registry["tls"]["cert_file"]
        key = registry["tls"]["key_file"]

    if config_version == 3:
        hosts = await containerd_registry_config(unit, host)
        mirror = next(iter(hosts["host"].values()))
        ca = mirror["ca"]
        cert, key = mirror["client"][0]

    async def file_content(path):
        output = await JujuRun.command(unit, f"base64 -w0 {shlex.quote(path)}")
        return output.stdout

    config = {
        "url": f"https://{host}",
        "ca_file": await file_content(ca),
        "cert_file": await file_content(cert),
        "key_file": await file_content(key),
    }
    return config


async def test_containerd_install(ops_test):
    """Check apt package binary ownership."""
    for unit in ops_test.model.applications["containerd"].units:
        output = await JujuRun.command(unit, "dpkg --verify containerd")
        assert not output.stdout

        output = await JujuRun.command(unit, "test ! -e /usr/bin/containerd-shim")
        assert output.success


async def test_cri_version(ops_test):
    """Check ctr can connect to containerd daemon."""
    for unit in ops_test.model.applications["containerd"].units:
        output = await JujuRun.command(unit, "ctr version")
        assert output.success


async def test_cri_plugins(ops_test):
    """Check the status of CRI plugins."""
    expected = {
        1: {("io.containerd.grpc.v1", "cri")},
        2: {
            ("io.containerd.grpc.v1", "cri"),
            ("io.containerd.cri.v1", "images"),
            ("io.containerd.cri.v1", "runtime"),
        },
    }
    for unit in ops_test.model.applications["containerd"].units:
        major = await containerd_major_version(unit)
        output = await JujuRun.command(unit, "ctr plugins ls")
        healthy_plugins = set()
        for line in output.stdout.splitlines():
            fields = line.split()
            if len(fields) != 4:
                continue

            plugin_type, plugin_id, _, status = fields
            if status == "ok":
                healthy_plugins.add((plugin_type, plugin_id))

        assert expected[major] <= healthy_plugins


async def test_config_version_ignored(ops_test, juju_config):
    """Check that the deprecated config_version is ignored."""
    units = ops_test.model.applications["containerd"].units
    config = await containerd_config(units[0])
    config_version = containerd_config_version(config)
    wrong_version = "v2" if config_version == 1 else "v1"

    await juju_config("containerd", config_version=wrong_version)
    for unit in units:
        config = await containerd_config(unit)
        assert containerd_config_version(config) == config_version


async def test_config_file_permissions(ops_test):
    """Check containerd config files are only readable by root."""
    # Certificate files under /root/cdk are managed by the principal charm.
    for unit in ops_test.model.applications["containerd"].units:
        config = await containerd_config(unit)
        paths = ["/etc/containerd/config.toml"]
        if containerd_config_version(config) == 3:
            output = await JujuRun.command(
                unit,
                "find /etc/containerd/certs.d -name hosts.toml -type f",
            )
            paths.extend(output.stdout.splitlines())

        command = "stat -c '%a' " + " ".join(shlex.quote(path) for path in paths)
        output = await JujuRun.command(unit, command)
        assert set(output.stdout.splitlines()) == {"600"}


async def test_config_dockerio_registry_exists(ops_test):
    """Check the Docker Hub registry exists in containerd config."""
    for unit in ops_test.model.applications["containerd"].units:
        config = await containerd_config(unit)
        config_version = containerd_config_version(config)

        if config_version == 1:
            mirrors = config["plugins"]["cri"]["registry"]["mirrors"]
        if config_version == 2:
            mirrors = config["plugins"]["io.containerd.grpc.v1.cri"]["registry"]["mirrors"]
        if config_version in (1, 2):
            assert "docker.io" in mirrors, "docker.io missing from containerd config"
            assert mirrors["docker.io"]["endpoint"] == ["https://registry-1.docker.io"]

        if config_version == 3:
            hosts = await containerd_registry_config(unit, "docker.io")
            assert hosts["server"] == "https://docker.io"
            assert "https://registry-1.docker.io" in hosts["host"]


async def test_config_relation_registry_exists(ops_test):
    """Check the relation registry exists in containerd config."""
    registry_unit = ops_test.model.applications["docker-registry"].units[0]
    for unit in ops_test.model.applications["containerd"].units:
        host = registry_host(ops_test)
        config = await containerd_config(unit)
        config_version = containerd_config_version(config)

        if config_version == 1:
            configs = config["plugins"]["cri"]["registry"]["configs"]
        if config_version == 2:
            configs = config["plugins"]["io.containerd.grpc.v1.cri"]["registry"]["configs"]
        if config_version in (1, 2):
            assert len(configs) == 1, "registry config isn't represented in config.toml"
            docker_registry = next(iter(configs))
            assert configs[docker_registry]["tls"], "TLS config isn't represented in the config.toml"
            assert docker_registry in registry_unit.workload_status_message

        if config_version == 3:
            hosts = await containerd_registry_config(unit, host)
            mirror = next(iter(hosts["host"].values()))
            assert mirror["ca"], "CA config isn't represented in hosts.toml"
            assert mirror["client"], "client TLS config isn't represented in hosts.toml"

            registry = config["plugins"]["io.containerd.cri.v1.images"]["registry"]
            assert "configs" not in registry


async def test_proxy_registry_pull(ops_test, juju_config):
    """Check containerd uses its configured proxy."""
    unit = ops_test.model.applications["containerd"].units[0]
    service = "containerd-proxy-test"
    port = 18080
    proxy = {
        "http_proxy": "",
        "https_proxy": f"http://127.0.0.1:{port}",
        "no_proxy": "",
    }

    await JujuRun.command(unit, f"systemctl stop {service}", check=False)
    start = f"systemd-run --unit={service} --collect --quiet " f"python3 -m http.server {port} --bind 127.0.0.1"
    await JujuRun.command(unit, start)

    try:
        await juju_config("containerd", **proxy)
        pull = await JujuRun.command(unit, "ctr images pull proxy-test.invalid/test:latest", check=False)
        assert not pull.success

        logs = await JujuRun.command(unit, f"journalctl --unit={service} --no-pager --output=cat")
        assert "CONNECT proxy-test.invalid:443" in logs.stdout
    finally:
        await JujuRun.command(unit, f"systemctl stop {service}", check=False)
        containerd = ops_test.model.applications["containerd"]
        await containerd.set_config({key: "" for key in proxy})
        await ops_test.model.wait_for_idle(apps=["containerd"], status="active", timeout=10 * 60)


async def test_public_registry_pull(ops_test, public_registry_image):
    """Pull an image from a public registry."""
    unit = ops_test.model.applications["containerd"].units[0]
    await run_image(unit, "containerd-public-pull", public_registry_image)


async def test_private_registry_tls_pull(ops_test, private_registry_image):
    """Pull an image from the private TLS registry."""
    unit = ops_test.model.applications["containerd"].units[0]
    await run_image(unit, "containerd-private-tls-pull", private_registry_image)


async def test_private_registry_basic_auth_pull(ops_test, juju_config, private_registry_image):
    """Pull an image from the private registry with basic auth."""
    unit = ops_test.model.applications["containerd"].units[0]
    username = "admin"
    password = f"integration-test-{uuid.uuid4().hex}"
    await juju_config(
        "docker-registry",
        **{
            "auth-basic-user": username,
            "auth-basic-password": password,
        },
    )
    await ops_test.model.wait_for_idle(apps=["containerd"], status="active", timeout=10 * 60)

    await run_image(unit, "containerd-private-auth-pull", private_registry_image)


@pytest.mark.parametrize(
    "invalid_config",
    ("", "{}", "[{}]"),
    ids=("empty", "not-a-list", "missing-url"),
)
async def test_custom_registries_invalid(ops_test, invalid_config):
    """Check invalid registry configuration blocks the charm and can be corrected."""
    containerd = ops_test.model.applications["containerd"]
    config = await containerd.get_config()
    previous = config["custom_registries"]["value"]

    try:
        await containerd.set_config({"custom_registries": invalid_config})
        await ops_test.model.wait_for_idle(
            apps=["containerd"],
            status="blocked",
            timeout=10 * 60,
        )
        for unit in containerd.units:
            assert unit.workload_status_message == "Invalid custom_registries configuration"
    finally:
        await containerd.set_config({"custom_registries": previous})
        await ops_test.model.wait_for_idle(
            apps=["containerd"],
            status="active",
            timeout=10 * 60,
        )


async def test_custom_registry_tls_pull(
    ops_test,
    custom_registry_image,
    custom_registry_config,
):
    """Pull an image from a custom TLS registry."""
    unit = ops_test.model.applications["containerd"].units[0]
    containerd = ops_test.model.applications["containerd"]

    try:
        await containerd.set_config({"custom_registries": json.dumps([custom_registry_config])})
        await ops_test.model.wait_for_idle(apps=["containerd"], status="active", timeout=10 * 60)
        await run_image(unit, "containerd-custom-tls-pull", custom_registry_image)
    finally:
        await containerd.set_config({"custom_registries": "[]"})
        await ops_test.model.wait_for_idle(apps=["containerd"], status="active", timeout=10 * 60)


async def test_custom_registry_basic_auth_pull(
    ops_test,
    juju_config,
    custom_registry_image,
    custom_registry_config,
):
    """Pull an image from a custom registry with basic auth."""
    unit = ops_test.model.applications["containerd"].units[0]
    username = "admin"
    password = f"integration-test-{uuid.uuid4().hex}"
    await juju_config(
        "custom-registry",
        **{
            "auth-basic-user": username,
            "auth-basic-password": password,
        },
    )
    custom_registry_config.update({"username": username, "password": password})
    containerd = ops_test.model.applications["containerd"]

    try:
        await containerd.set_config({"custom_registries": json.dumps([custom_registry_config])})
        await ops_test.model.wait_for_idle(apps=["containerd"], status="active", timeout=10 * 60)
        await run_image(unit, "containerd-custom-auth-pull", custom_registry_image)
    finally:
        await containerd.set_config({"custom_registries": "[]"})
        await ops_test.model.wait_for_idle(apps=["containerd"], status="active", timeout=10 * 60)


async def test_private_registry_relation_cleanup(ops_test, juju_config):
    """Check remove relation cleans up the registry config."""
    await juju_config("docker-registry", **{"auth-basic-password": "integration-test-password"})
    await ops_test.model.wait_for_idle(apps=["containerd"], status="active", timeout=10 * 60)

    host = registry_host(ops_test)
    await ops_test.juju(
        "remove-relation",
        "docker-registry:docker-registry",
        "containerd:docker-registry",
        check=True,
    )
    try:
        await ops_test.model.wait_for_idle(apps=["containerd"], status="active", timeout=10 * 60)
        for unit in ops_test.model.applications["containerd"].units:
            config = await containerd_config(unit)
            config_version = containerd_config_version(config)

            if config_version == 1:
                registry = config["plugins"]["cri"]["registry"]
            if config_version == 2:
                registry = config["plugins"]["io.containerd.grpc.v1.cri"]["registry"]
            if config_version in (1, 2):
                configs = registry.get("configs", {})
                assert all(host not in key for key in configs)

            if config_version == 3:
                path = f"/etc/containerd/certs.d/{host}/hosts.toml"
                output = await JujuRun.command(unit, f"test ! -e {shlex.quote(path)}")
                assert output.success
                registry = config["plugins"]["io.containerd.cri.v1.images"]["registry"]
                configs = registry.get("configs", {})
                assert all(host not in key for key in configs)
    finally:
        await ops_test.model.add_relation(
            "docker-registry:docker-registry",
            "containerd:docker-registry",
        )
        await ops_test.model.wait_for_idle(
            apps=["containerd", "docker-registry"],
            status="active",
            timeout=10 * 60,
        )


async def test_containerd_disable_gpu_support(ops_test, juju_config):
    """Test that disabling gpu support removes nvidia drivers."""
    await juju_config("containerd", gpu_driver="none")
    for unit in ops_test.model.applications["containerd"].units:
        output = await JujuRun.command(unit, "cat /etc/apt/sources.list.d/nvidia.list", check=False)
        assert "No such file " in output.stderr, "NVIDIA sources list was populated"

        output = await JujuRun.command(unit, "dpkg-query --list cuda-drivers", check=False)
        assert "cuda-drivers" in output.stderr, "cuda-drivers shouldn't be installed"


async def test_containerd_nvidia_gpu_support(ops_test, juju_config):
    """Test that enabling gpu support installed nvidia drivers."""
    await juju_config("containerd", gpu_driver="nvidia", _timeout=15 * 60)
    for unit in ops_test.model.applications["containerd"].units:
        output = await JujuRun.command(unit, "cat /etc/apt/sources.list.d/nvidia.list")
        assert output.stdout, "NVIDIA sources list was empty"

        output = await JujuRun.command(unit, "dpkg-query --list cuda-drivers")
        assert "cuda-drivers" in output.stdout, "cuda-drivers not installed"


@pytest.mark.xfail(reason="No apt repo for nvidia-container-runtime")
async def test_upgrade_action_gpu_force(ops_test):
    """Test running upgrade action with GPU and force."""
    unit = ops_test.model.applications["containerd"].units[0]
    start = await process_elapsed_time(unit, "containerd")
    action = await JujuRun.action(unit, "upgrade-packages", containerd=False, gpu=True, force=True)
    results = action.results
    log.info(f"Upgrade results = '{results}'")
    assert results["cuda-drivers"]["available"] == results["cuda-drivers"]["installed"]
    assert results["cuda-drivers"]["upgrade-available"] == "False"
    assert results["cuda-drivers"]["upgrade-complete"] == "True"
    end = await process_elapsed_time(unit, "containerd")
    assert end >= start, "containerd service shouldn't have been restarted"


@pytest_asyncio.fixture()
async def microbots(ops_test: OpsTest, tmp_path: Path):
    """Start microbots workload on each k8s-worker, cleanup at the end of the test."""
    workers: Application = ops_test.model.applications["kubernetes-worker"]
    any_worker: Unit = workers.units[0]
    arch = any_worker.machine.safe_data["hardware-characteristics"]["arch"]

    context = {
        "public_address": any_worker.public_address,
        "replicas": len(workers.units),
        "arch": arch,
    }
    rendered = str(tmp_path / "microbot.yaml")
    microbot = jinja2.Template(Path("tests/data/microbot.yaml.j2").read_text())
    microbot.stream(**context).dump(rendered)
    apply = format_kubectl_cmd("apply -f /tmp/microbot.yaml")
    delete = format_kubectl_cmd("delete -f /tmp/microbot.yaml --ignore-not-found")
    cleanup = format_kubectl_cmd("delete pod -l=app=microbot --ignore-not-found --force --grace-period=0")
    try:
        cmd = f"scp {rendered} {any_worker.name}:/tmp/microbot.yaml"
        await ops_test.juju(*shlex.split(cmd), check=True)

        # Remove resources left by an interrupted run before recreating them.
        await JujuRun.command(any_worker, cleanup, check=False)

        await JujuRun.command(any_worker, apply)
        pods = await pods_in_state(any_worker, {"app": "microbot"}, "Running")
        yield len(pods)
    finally:
        await JujuRun.command(any_worker, delete)


async def test_restart_containerd(microbots, ops_test: OpsTest):
    """Test microbots continue running while containerd stopped."""
    containerds = ops_test.model.applications["containerd"]
    num_units = len(containerds.units)
    any_containerd = containerds.units[0]
    try:
        await asyncio.gather(*(JujuRun.command(_, "service containerd stop") for _ in containerds.units))
        async with ops_test.fast_forward():
            await ops_test.model.wait_for_idle(apps=["containerd"], status="blocked", timeout=6 * 60)

        await nodes_in_state(any_containerd, num_units, "NotReady")

        # test that pods are still running while containerd is offline
        pods = await JujuRun.command(any_containerd, format_kubectl_cmd("get pods -l=app=microbot"))
        assert pods.stdout.count("microbot") == microbots, f"Ensure {microbots} pod(s) are installed"
        assert pods.stdout.count("Running") == microbots, f"Ensure {microbots} pod(s) are running with containerd down"

        cluster_ip = await JujuRun.command(
            any_containerd,
            format_kubectl_cmd("get service -l=app=microbot -ojsonpath='{.items[*].spec.clusterIP}'"),
        )
        endpoint = f"http://{cluster_ip.stdout.strip()}"
        await JujuRun.command(any_containerd, f"curl {endpoint}")
    finally:
        await asyncio.gather(*(JujuRun.command(_, "service containerd start") for _ in containerds.units))
        async with ops_test.fast_forward():
            await ops_test.model.wait_for_idle(apps=["containerd"], status="active", timeout=6 * 60)


async def test_resource_containerd_migration(ops_test: OpsTest):
    """Replace legacy resource binaries with the apt package."""
    # TODO: Consider removing this test if unreliable.
    charm = next(Path.cwd().glob("containerd*.charm"), None)
    if not charm:
        charm = await ops_test.build_charm(".")
    charm = charm.resolve()

    await ops_test.track_model(
        "resource-migration",
        keep=False,
        config={"default-base": "ubuntu@24.04"},
    )
    try:
        with ops_test.model_context("resource-migration") as model:
            await model.deploy(
                "kubernetes-worker",
                channel="latest/edge",
                base="ubuntu@24.04",
            )
            # Revision 104 installs containerd from its attached resource.
            await model.deploy(
                "containerd",
                application_name="containerd",
                channel="latest/edge",
                revision=104,
                base="ubuntu@24.04",
            )
            await model.add_relation(
                "containerd:containerd",
                "kubernetes-worker:container-runtime",
            )
            # The old charm may fail after downgrading apt containerd.
            # The resource only needs to be installed.
            await model.wait_for_idle(
                apps=["containerd"],
                raise_on_error=False,
                timeout=20 * 60,
            )

            # Confirm the old charm replaced apt containerd with resource binaries.
            unit = model.applications["containerd"].units[0]
            resource_version = await containerd_version(unit)
            assert resource_version.startswith("1.")

            output = await JujuRun.command(unit, "test -x /usr/bin/containerd-shim")
            assert output.success

            # Refresh the deployed old charm to the local charm.
            await model.applications["containerd"].refresh(
                path=charm,
                force_units=True,
            )
            await model.wait_for_idle(
                apps=["containerd"],
                status="active",
                timeout=20 * 60,
            )

            # Confirm the local charm replaced resource binaries with apt binaries.
            unit = model.applications["containerd"].units[0]
            apt_version = await containerd_version(unit)
            assert apt_version != resource_version

            output = await JujuRun.command(unit, "dpkg --verify containerd")
            assert not output.stdout

            output = await JujuRun.command(unit, "test ! -e /usr/bin/containerd-shim")
            assert output.success

            output = await JujuRun.command(unit, "ctr version")
            assert output.success

            major = await containerd_major_version(unit)
            expected_plugins = {
                1: {("io.containerd.grpc.v1", "cri")},
                2: {
                    ("io.containerd.grpc.v1", "cri"),
                    ("io.containerd.cri.v1", "images"),
                    ("io.containerd.cri.v1", "runtime"),
                },
            }
            output = await JujuRun.command(unit, "ctr plugins ls")
            healthy_plugins = set()
            for line in output.stdout.splitlines():
                fields = line.split()
                if len(fields) != 4:
                    continue

                plugin_type, plugin_id, _, status = fields
                if status == "ok":
                    healthy_plugins.add((plugin_type, plugin_id))
            assert expected_plugins[major] <= healthy_plugins
            assert unit.workload_status_message == "Container runtime available"
    finally:
        # NOTE: May emit "Task was destroyed but it is pending" warnings here.
        await ops_test.forget_model(
            "resource-migration",
            timeout=20 * 60,
            destroy_storage=True,
            allow_failure=False,
        )
