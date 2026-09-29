import pathlib
import os
import json
from subprocess import CalledProcessError, STDOUT
import unittest.mock as mock
from urllib.error import HTTPError
import yaml

from charmhelpers.core import unitdata, host
from charmhelpers.core.templating import render
from charmhelpers.fetch import import_key
from charms.reactive import is_state, set_state
from reactive import containerd
import tempfile
import pytest

import jinja2


def test_series_upgrade():
    """Verify series upgrade hook sets the status."""
    flags = {
        "upgrade.series.in-progress": True,
        "containerd.nvidia.invalid-option": False,
    }
    is_state.side_effect = lambda flag: flags[flag]
    assert containerd.status.blocked.call_count == 0
    with mock.patch("reactive.containerd._check_containerd", return_value=False):
        containerd.charm_status()
    containerd.status.blocked.assert_called_once_with("Series upgrade in progress")


@pytest.mark.parametrize(
    "version,plugins",
    [
        (
            (1, 7, 36),
            b"TYPE ID PLATFORMS STATUS\nio.containerd.grpc.v1 cri linux/amd64 ok\n",
        ),
        (
            (2, 4, 1),
            (
                b"TYPE ID PLATFORMS STATUS\n"
                b"io.containerd.cri.v1 images - ok\n"
                b"io.containerd.cri.v1 runtime linux/amd64 ok\n"
                b"io.containerd.grpc.v1 cri linux/amd64 ok\n"
            ),
        ),
    ],
)
@mock.patch.object(containerd, "check_output")
def test_check_containerd_healthy(check_output, version, plugins):
    """Check healthy CRI plugins for each config version."""
    version_output = "Version: {}".format(".".join(map(str, version))).encode()
    check_output.side_effect = [version_output, b"", plugins]

    assert containerd._check_containerd()
    assert check_output.call_args_list == [
        mock.call(["containerd", "--version"]),
        mock.call(["ctr", "version"]),
        mock.call(["ctr", "plugins", "ls"]),
    ]


@pytest.mark.parametrize(
    "version,plugins",
    [
        (
            (1, 7, 36),
            b"TYPE ID PLATFORMS STATUS\nio.containerd.grpc.v1 cri linux/amd64 error\n",
        ),
        (
            (2, 4, 1),
            (
                b"TYPE ID PLATFORMS STATUS\n"
                b"io.containerd.cri.v1 images - ok\n"
                b"io.containerd.cri.v1 runtime linux/amd64 error\n"
                b"io.containerd.grpc.v1 cri linux/amd64 ok\n"
            ),
        ),
    ],
)
@mock.patch.object(containerd, "check_output")
def test_check_containerd_unhealthy(check_output, version, plugins):
    """Check unhealthy CRI plugins for each config version."""
    version_output = "Version: {}".format(".".join(map(str, version))).encode()
    check_output.side_effect = [version_output, b"", plugins]

    assert not containerd._check_containerd()


@mock.patch.object(containerd, "config_changed")
@mock.patch.object(containerd, "endpoint_from_flag")
@mock.patch.object(containerd, "ca_crt_path")
@mock.patch.object(containerd, "server_crt_path")
@mock.patch.object(containerd, "server_key_path")
def test_registry_relation(server_key_path, server_crt_path, ca_crt_path, endpoint_from_flag, config_changed):
    """Verify writing to the registry db keyvalue store."""
    mock_registry = endpoint_from_flag.return_value
    mock_registry.registry_netloc = "http://registry.relation:5000"

    mock_registry.has_auth_basic.return_value = True
    mock_registry.basic_user = "user"
    mock_registry.basic_password = "password"

    mock_registry.has_tls.return_value = True

    ca_crt_path.__str__.return_value = "/path/to/ca"
    server_crt_path.__str__.return_value = "/path/to/crt"
    server_key_path.__str__.return_value = "/path/to/key"

    containerd.configure_registry()

    config_changed.assert_called_once_with()
    set_registry_data = unitdata.kv().get("registry")
    assert set_registry_data == {
        "url": "http://registry.relation:5000",
        "host": "registry.relation:5000",
        "username": "user",
        "password": "password",
        "ca_file": None,
        "cert_file": None,
        "key_file": None,
        "insecure_skip_verify": None,
        "ca": "/path/to/ca",
        "cert": "/path/to/crt",
        "key": "/path/to/key",
    }


@pytest.mark.parametrize(
    "registry_errors",
    [
        ("", "Failed to decode json string"),
        ("{}", "custom_registries is not a list"),
        ("[1]", "registry #0 is not in object form"),
        ("[{}]", "registry #0 missing required field 'url'"),
        ('[{"url": 1}]', "registry #0 field url=1 is type int, not type str"),
        (
            '[{"url": "", "insecure_skip_verify": "FALSE"}]',
            "registry #0 field insecure_skip_verify=FALSE is type str, not type bool",
        ),
        (
            '[{"url": "", "why-am-i-here": "abc"}]',
            "registry #0 field why-am-i-here may not be specified",
        ),
        (
            '[{"url": "https://docker.io"}, {"url": "https://docker.io"}]',
            "registry #1 defines docker.io more than once",
        ),
        ("[]", None),
    ],
    ids=[
        "Invalid JSON",
        "Not a List",
        "List Item not an object",
        "Missing required field",
        "Non-stringly typed field",
        "Accidentally truthy",
        "Restricted field",
        "Duplicate host",
        "No errors",
    ],
)
def test_invalid_custom_registries(registry_errors):
    """Verify error status for invalid custom registries configurations."""
    registries, expected = registry_errors
    actual = containerd.invalid_custom_registries(registries)
    assert actual == expected


def test_registries_list():
    """Verify _registries_list resolves json to a list of objects, or returns default."""
    assert containerd._registries_list("[]") == []

    default = []
    assert containerd._registries_list("[{]", default) is default, "return default when invalid json"
    assert containerd._registries_list("{}", default) is default, "return default when valid json isn't a list"

    with pytest.raises(containerd.ValidationError) as ie:
        containerd._registries_list("[{]")
    assert "Failed to decode json string" in str(ie.value)

    with pytest.raises(containerd.ValidationError) as ie:
        containerd._registries_list("{}")
    assert "is not a list" in str(ie.value)


def test_merge_custom_registries(tmp_path):
    """Verify merges of registries."""
    config = [
        {"url": "my.registry:port", "username": "user", "password": "pass"},
        {
            "url": "my.other.registry",
            "ca_file": "aGVsbG8gd29ybGQgY2EtZmlsZQ==",
            "key_file": "aGVsbG8gd29ybGQga2V5LWZpbGU=",
            "cert_file": "abc",  # invalid base64 is ignored
        },
    ]
    ctxs = containerd.merge_custom_registries(tmp_path, json.dumps(config), None)
    with open(os.path.join(tmp_path, "my.other.registry.ca")) as f:
        assert f.read() == "hello world ca-file"
    with open(os.path.join(tmp_path, "my.other.registry.key")) as f:
        assert f.read() == "hello world key-file"
    assert not os.path.exists(os.path.join(tmp_path, "my.other.registry.cert"))

    for ctx in ctxs:
        assert ctx.url, "url must be assigned"

    # Remove 'my.other.registry' from config
    new_config = [{"url": "my.registry:port", "username": "user", "password": "pass"}]
    ctxs = containerd.merge_custom_registries(tmp_path, json.dumps(new_config), json.dumps(config))
    assert not os.path.exists(os.path.join(tmp_path, "my.other.registry.ca"))
    assert not os.path.exists(os.path.join(tmp_path, "my.other.registry.key"))
    assert not os.path.exists(os.path.join(tmp_path, "my.other.registry.cert"))


def test_merge_custom_registries_duplicate(tmp_path):
    """Reject duplicate relation and config registries."""
    unitdata.kv().set(
        "registry",
        {
            "url": "https://mirror.example",
            "host": "registry.example",
        },
    )
    registries = json.dumps([{"url": "https://registry.example"}])

    with pytest.raises(containerd.DuplicateError):
        containerd.merge_custom_registries(tmp_path, registries, None)


def test_insert_registry_from_relation():
    """Add a relation registry."""
    relation_registry = {
        "url": "https://mirror.example",
        "host": "registry.example",
    }
    unitdata.kv().set("registry", relation_registry)
    registries = [containerd.Registry(url="https://other.example")]

    result = containerd._insert_registry_from_relation(registries)

    assert [registry.host for registry in result] == [
        "other.example",
        "registry.example",
    ]


def test_insert_registry_from_relation_duplicate():
    """Reject a duplicate relation registry."""
    relation_registry = {
        "url": "https://mirror.example",
        "host": "registry.example",
    }
    unitdata.kv().set("registry", relation_registry)
    registries = [containerd.Registry(url="https://registry.example")]

    with pytest.raises(containerd.DuplicateError):
        containerd._insert_registry_from_relation(registries)


@mock.patch.object(containerd, "render")
def test_render_registry_config(render, tmp_path):
    """Render and track registry hosts."""
    registry = containerd.Registry(
        url="https://mirror.example",
        host="registry.example",
    )

    containerd._render_registry_config(str(tmp_path), [registry])

    render.assert_called_once_with(
        "hosts.toml",
        str(tmp_path / "certs.d" / "registry.example" / "hosts.toml"),
        {"registry": registry},
        perms=0o600,
    )
    assert unitdata.kv().get("registry-hosts") == ["registry.example"]


@mock.patch.object(containerd, "render")
def test_render_registry_config_stale(render, tmp_path):
    """Remove stale registry hosts."""
    stale_directory = tmp_path / "certs.d" / "stale.example"
    stale_directory.mkdir(parents=True)
    (stale_directory / "hosts.toml").write_text("stale")
    unitdata.kv().set("registry-hosts", ["stale.example"])
    registry = containerd.Registry(url="https://mirror.example", host="registry.example")

    containerd._render_registry_config(str(tmp_path), [registry])

    render.assert_called_once_with(
        "hosts.toml",
        str(tmp_path / "certs.d" / "registry.example" / "hosts.toml"),
        {"registry": registry},
        perms=0o600,
    )
    assert not stale_directory.exists()
    assert unitdata.kv().get("registry-hosts") == ["registry.example"]


def test_registry_hosts_config():
    """Render mirror, fallback, and TLS settings."""
    registry = containerd.Registry(
        url="http://mirror.example:5000",
        host="registry.example",
        insecure_skip_verify=True,
    )
    registry.ca = "/path/to/ca"
    registry.cert = "/path/to/cert"
    registry.key = "/path/to/key"
    env = jinja2.Environment(loader=jinja2.FileSystemLoader("src/templates"))

    output = env.get_template("hosts.toml").render(registry=registry)

    assert 'server = "https://registry.example"' in output
    assert '[host."http://mirror.example:5000"]' in output
    assert 'ca = "/path/to/ca"' in output
    assert 'client = [["/path/to/cert", "/path/to/key"]]' in output
    assert "skip_verify = true" in output


@pytest.mark.parametrize("gpu", ("off", "on"), ids=("gpu off", "gpu on"))
@mock.patch("reactive.containerd._installed_containerd_version")
@mock.patch("reactive.containerd.endpoint_from_flag")
@mock.patch("reactive.containerd.config")
@mock.patch("charms.layer.containerd.can_mount_cgroup2", mock.Mock(return_value=False))
def test_render_config_v2(
    mock_config,
    mock_endpoint_from_flag,
    mock_installed_containerd_version,
    gpu,
    tmp_path,
):
    """Render the containerd 1.x config."""

    class MockConfig(dict):
        def changed(self, *_args, **_kwargs):
            return False

    def jinja_render(source, target, context, **_kwargs):
        env = jinja2.Environment(loader=jinja2.FileSystemLoader("src/templates"))
        template = env.get_template(source)
        pathlib.Path(target).parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as fp:
            fp.write(template.render(context))

    render.side_effect = jinja_render
    config = mock_config.return_value = MockConfig(config_version="v1", gpu_driver="auto", runtime="auto")
    mock_installed_containerd_version.return_value = (1, 7, 36)
    mock_endpoint_from_flag.return_value.get_sandbox_image.return_value = "sandbox-image"
    flags = {
        "containerd.nvidia.available": gpu == "on",
    }
    is_state.side_effect = lambda flag: flags[flag]
    config["custom_registries"] = json.dumps(
        [
            {"url": "my.registry:port", "username": "user", "password": {"interesting": "json"}},
            {"url": "my.other.registry", "insecure_skip_verify": True},
        ]
    )
    unitdata.kv().set(
        "registry",
        {
            "url": "http://db.registry:5000",
            "username": "user",
            "password": "pass",
            "ca": "/known/file/path/ca.crt",
            "cert": "/known/file/path/cert.crt",
            "key": "/known/file/path/cert.key",
        },
    )
    with mock.patch("reactive.containerd.CONFIG_DIRECTORY", tmp_path):
        containerd.config_changed()

    target = pathlib.Path(tmp_path) / "config.toml"
    expected = pathlib.Path(__file__).parent / "test_render_config_v2" / f"nvidia-{gpu}-v2-config.toml"
    assert target.read_text() == expected.read_text()


@pytest.mark.parametrize("gpu", ("off", "on"), ids=("gpu off", "gpu on"))
@mock.patch("reactive.containerd._installed_containerd_version")
@mock.patch("reactive.containerd.endpoint_from_flag")
@mock.patch("reactive.containerd.config")
def test_render_config_v3(
    mock_config,
    mock_endpoint_from_flag,
    mock_installed_containerd_version,
    gpu,
    tmp_path,
):
    """Render the containerd 2.x config."""

    class MockConfig(dict):
        def changed(self, *_args, **_kwargs):
            return False

    def jinja_render(source, target, context, **_kwargs):
        env = jinja2.Environment(loader=jinja2.FileSystemLoader("src/templates"))
        template = env.get_template(source)
        pathlib.Path(target).parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as fp:
            fp.write(template.render(context))

    render.side_effect = jinja_render
    config = mock_config.return_value = MockConfig(config_version="v1", gpu_driver="auto", runtime="auto")
    mock_installed_containerd_version.return_value = (2, 4, 1)
    mock_endpoint_from_flag.return_value.get_sandbox_image.return_value = "sandbox-image"
    flags = {
        "containerd.nvidia.available": gpu == "on",
    }
    is_state.side_effect = lambda flag: flags[flag]
    config["custom_registries"] = json.dumps(
        [
            {"url": "my.registry:port", "username": "user", "password": {"interesting": "json"}},
            {"url": "my.other.registry", "insecure_skip_verify": True},
        ]
    )
    unitdata.kv().set(
        "registry",
        {
            "url": "http://db.registry:5000",
            "username": "user",
            "password": "pass",
            "ca": "/known/file/path/ca.crt",
            "cert": "/known/file/path/cert.crt",
            "key": "/known/file/path/cert.key",
        },
    )
    with mock.patch("reactive.containerd.CONFIG_DIRECTORY", tmp_path):
        containerd.config_changed()

    expected_directory = pathlib.Path(__file__).parent / "test_render_config_v3"
    target = pathlib.Path(tmp_path) / "config.toml"
    expected = expected_directory / f"nvidia-{gpu}-config.toml"
    assert target.read_text() == expected.read_text().rstrip("\n")

    for registry_host in ("docker.io", "my.registry:port", "my.other.registry", "db.registry:5000"):
        target = pathlib.Path(tmp_path) / "certs.d" / registry_host / "hosts.toml"
        expected = expected_directory / registry_host / "hosts.toml"
        assert target.read_text() == expected.read_text().rstrip("\n")


@mock.patch.object(containerd, "_check_containerd", return_value=True)
@mock.patch.object(containerd.host, "service_restart", return_value=True)
@mock.patch.object(containerd, "_render_config", return_value=True)
@mock.patch.object(containerd, "apt_install")
@mock.patch.object(containerd, "apt_unhold")
@mock.patch.object(containerd, "apt_hold")
def test_reinstall_containerd(
    apt_hold,
    apt_unhold,
    apt_install,
    render_config,
    service_restart,
    check_containerd,
):
    """Render candidate config before apt runs."""
    calls = mock.Mock()
    for name, mocked in (
        ("render_config", render_config),
        ("apt_unhold", apt_unhold),
        ("apt_install", apt_install),
        ("apt_hold", apt_hold),
        ("service_restart", service_restart),
        ("check_containerd", check_containerd),
    ):
        calls.attach_mock(mocked, name)

    containerd.reinstall_containerd((2, 4, 1))

    # Check the full call order.
    assert calls.mock_calls == [
        mock.call.render_config(version=(2, 4, 1)),
        mock.call.apt_unhold(containerd.CONTAINERD_PACKAGE),
        mock.call.apt_install([containerd.CONTAINERD_PACKAGE, "--reinstall"], fatal=True),
        mock.call.apt_hold(containerd.CONTAINERD_PACKAGE),
        mock.call.service_restart("containerd.service"),
        mock.call.check_containerd(),
    ]


@mock.patch.object(containerd, "_check_containerd", return_value=True)
@mock.patch.object(containerd.host, "service_restart", return_value=True)
@mock.patch.object(containerd, "_render_config", return_value=True)
@mock.patch.object(containerd, "_installed_containerd_version", return_value=(1, 7, 36))
@mock.patch.object(containerd, "apt_install", side_effect=CalledProcessError(1, "apt-get"))
@mock.patch.object(containerd, "apt_unhold")
@mock.patch.object(containerd, "apt_hold")
def test_reinstall_containerd_apt_failure(
    apt_hold,
    apt_unhold,
    apt_install,
    installed_containerd_version,
    render_config,
    service_restart,
    check_containerd,
):
    """Restore config after an apt failure."""
    calls = mock.Mock()
    for name, mocked in (
        ("render_config", render_config),
        ("apt_unhold", apt_unhold),
        ("apt_install", apt_install),
        ("installed_containerd_version", installed_containerd_version),
        ("service_restart", service_restart),
        ("check_containerd", check_containerd),
        ("apt_hold", apt_hold),
    ):
        calls.attach_mock(mocked, name)

    with pytest.raises(CalledProcessError):
        containerd.reinstall_containerd((2, 4, 1))

    assert calls.mock_calls == [
        mock.call.render_config(version=(2, 4, 1)),
        mock.call.apt_unhold(containerd.CONTAINERD_PACKAGE),
        mock.call.apt_install([containerd.CONTAINERD_PACKAGE, "--reinstall"], fatal=True),
        mock.call.installed_containerd_version(),
        mock.call.render_config(version=(1, 7, 36)),
        mock.call.service_restart("containerd.service"),
        mock.call.check_containerd(),
        mock.call.apt_hold(containerd.CONTAINERD_PACKAGE),
    ]


@mock.patch.object(containerd, "set_state")
@mock.patch.object(containerd, "remove_state")
@mock.patch.object(containerd, "check_output")
@mock.patch.object(containerd.os.path, "lexists")
@mock.patch.object(containerd.os, "remove")
@mock.patch.object(containerd, "reinstall_containerd")
@mock.patch.object(containerd, "candidate_containerd_version", return_value=(2, 4, 1))
@mock.patch.object(containerd, "_installed_containerd_version", return_value=(1, 7, 36))
@mock.patch.object(containerd, "apt_update")
def test_migrate_resource_containerd(
    apt_update,
    installed_containerd_version,
    candidate_containerd_version,
    reinstall_containerd,
    remove,
    lexists,
    check_output,
    remove_state,
    set_state,
):
    """Replace resource binaries with apt binaries."""

    def binary_exists(path):
        existing = {"containerd", "containerd-shim"}
        return os.path.basename(path) in existing

    def dpkg_query(command, **_kwargs):
        if command == ["dpkg-query", "--search", pathlib.Path("/usr/bin/containerd-shim")]:
            raise CalledProcessError(1, command)
        if command == ["dpkg-query", "--search", pathlib.Path("/usr/bin/containerd")]:
            return b"containerd: /usr/bin/containerd"
        raise AssertionError("Unexpected command: {}".format(command))

    lexists.side_effect = binary_exists
    check_output.side_effect = dpkg_query

    containerd.migrate_resource_containerd()

    apt_update.assert_called_once_with(fatal=True)
    installed_containerd_version.assert_called_once_with()
    candidate_containerd_version.assert_called_once_with()
    reinstall_containerd.assert_called_once_with((2, 4, 1))
    remove.assert_called_once_with(pathlib.Path("/usr/bin/containerd-shim"))
    remove_state.assert_has_calls(
        [
            mock.call("containerd.resource-migration.failed"),
            mock.call("containerd.resource.installed"),
            mock.call("containerd.resource.evaluated"),
        ]
    )
    set_state.assert_called_once_with("containerd.installed")


@mock.patch.object(containerd, "set_state")
@mock.patch.object(containerd, "remove_state")
@mock.patch.object(containerd, "reinstall_containerd")
@mock.patch.object(containerd, "candidate_containerd_version", return_value=(1, 6, 39))
@mock.patch.object(containerd, "_installed_containerd_version", return_value=(1, 7, 36))
@mock.patch.object(containerd, "apt_update")
def test_migrate_resource_containerd_downgrade(
    apt_update,
    installed_containerd_version,
    candidate_containerd_version,
    reinstall_containerd,
    remove_state,
    set_state,
):
    """Refuse an apt downgrade."""
    containerd.migrate_resource_containerd()

    reinstall_containerd.assert_not_called()
    remove_state.assert_not_called()
    set_state.assert_called_once_with("containerd.resource-migration.failed")


@mock.patch.object(containerd, "set_state")
@mock.patch.object(containerd, "remove_state")
@mock.patch.object(containerd, "reinstall_containerd", side_effect=CalledProcessError(1, "apt-get"))
@mock.patch.object(containerd, "candidate_containerd_version", return_value=(2, 4, 1))
@mock.patch.object(containerd, "_installed_containerd_version", return_value=(1, 7, 36))
@mock.patch.object(containerd, "apt_update")
def test_migrate_resource_containerd_failure(
    apt_update,
    installed_containerd_version,
    candidate_containerd_version,
    reinstall_containerd,
    remove_state,
    set_state,
):
    """Keep migration flags after a failure."""
    containerd.migrate_resource_containerd()

    remove_state.assert_not_called()
    set_state.assert_called_once_with("containerd.resource-migration.failed")


def test_juju_proxy_changed():
    """Verify proxy changed bools are set as expected."""
    cached = {"http_proxy": "foo", "https_proxy": "foo", "no_proxy": "foo"}
    new = {"http_proxy": "bar", "https_proxy": "bar", "no_proxy": "bar"}

    # Test when nothing is cached
    db = unitdata.kv()
    assert containerd._juju_proxy_changed() is True

    # Test when cache hasn't changed
    db.set("config-cache", cached)
    with mock.patch("reactive.containerd.check_for_juju_https_proxy", return_value=cached):
        assert containerd._juju_proxy_changed() is False

    # Test when cache has changed
    with mock.patch("reactive.containerd.check_for_juju_https_proxy", return_value=new):
        assert containerd._juju_proxy_changed() is True


@pytest.fixture()
def default_config():
    """Mock out the config method from the charm default config."""
    config_yaml = yaml.safe_load(pathlib.Path("src/config.yaml").read_bytes())
    values = {key: obj.get("default") for key, obj in config_yaml["options"].items()}
    with mock.patch.object(containerd, "config", side_effect=values.get) as obj:
        yield obj


@mock.patch.object(containerd, "env_proxy_settings")
@mock.patch.object(containerd, "log")
@pytest.mark.usefixtures("default_config")
@pytest.mark.parametrize("success", [True, False])
def test_fetch_url_text(log, env_proxy_settings, success):
    """Test the fetch url method for success and failures."""

    def _responder(*_args):
        if success:
            return response
        raise HTTPError(the_url, 404, "Not Found", [], None)

    env_proxy_settings.return_value = None
    the_url = "https://google.com/robots.txt"
    response = mock.MagicMock(autospec="urllib.client.HTTPResponse")
    response.status = 200
    with mock.patch("urllib.request.OpenerDirector.open", side_effect=_responder) as mock_open:
        text = containerd.fetch_url_text([the_url])
    env_proxy_settings.assert_called_once_with()
    mock_open.assert_called_once_with(the_url)
    if success:
        assert text == [response.read.return_value.decode.return_value]
        response.read.assert_called_once_with()
        response.read.return_value.decode.assert_called_once_with()
        log.assert_not_called()
    else:
        assert text == [None]
        response.read.assert_not_called()
        log.assert_called_once_with(f"Cannot fetch url='{the_url}' with code 404 Not Found")


@mock.patch.object(containerd, "config_changed")
@mock.patch.object(containerd, "apt_autoremove")
@mock.patch.object(os, "remove")
@mock.patch.object(containerd, "apt_purge")
@mock.patch("builtins.open")
@pytest.mark.usefixtures("default_config")
def test_unconfigure_nvidia(mock_open, mock_apt_purge, mock_os_remove, mock_apt_autoremove, mock_config_changed):
    """Verify NVIDIA config is removed."""
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = pathlib.Path(tmp_dir.name)
    sources_file = os.path.join(tmp_path, "nvidia.list")
    with mock.patch("reactive.containerd.NVIDIA_SOURCES_FILE", sources_file):
        containerd.unconfigure_nvidia()
    mock_apt_purge.assert_called_once
    mock_os_remove.assert_called_once
    mock_apt_autoremove.assert_called_once
    mock_config_changed.assert_called_once_with()
    assert not os.path.exists(sources_file)


@mock.patch.object(containerd, "fetch_url_text", return_value=["-key1-", "-key2-"])
@mock.patch("builtins.open")
@pytest.mark.usefixtures("default_config")
def test_configure_nvidia_sources(mock_open, fetch_url_text):
    """Verify NVIDIA apt sources are configured and keys are imported."""
    mock_lsb_release = dict(DISTRIB_ID="ubuntu", DISTRIB_RELEASE="20.04")
    import_key.reset_mock()
    with mock.patch.object(host, "lsb_release", return_value=mock_lsb_release):
        containerd.configure_nvidia_sources()

    # keys should be fetched from formatted urls
    fetch_url_text.assert_called_with(
        [
            "https://nvidia.github.io/nvidia-container-runtime/gpgkey",
            "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub",
        ]
    )

    # import_key should be called twice with two key responses
    assert import_key.call_count == 2
    import_key.assert_has_calls(
        [
            mock.call("-key1-"),
            mock.call("-key2-"),
        ]
    )

    # sources file should be written out
    mock_open.assert_called_once_with("/etc/apt/sources.list.d/nvidia.list", "w")
    mock_file = mock_open.return_value.__enter__()
    mock_file.write.assert_called_once_with(
        "deb https://nvidia.github.io/libnvidia-container/stable/deb/$(ARCH) /\n"
        "deb https://nvidia.github.io/nvidia-container-runtime/ubuntu20.04/$(ARCH) /\n"
        "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /"
    )


@mock.patch.object(containerd, "config_changed")
@mock.patch.object(containerd, "configure_nvidia_sources")
@mock.patch.object(containerd, "unconfigure_nvidia")
@mock.patch.object(containerd, "_test_gpu_reboot", mock.MagicMock())
@pytest.mark.usefixtures("default_config")
def test_install_nvidia_drivers(
    mock_unconfigure_nvidia,
    mock_configure_nvidia_sources,
    mock_config_changed,
):
    """Verify drivers are removed, config is done, and containerd config is updated."""
    set_state.reset_mock()
    containerd.install_nvidia_drivers()
    mock_unconfigure_nvidia.assert_called_once_with(reconfigure=False)
    mock_configure_nvidia_sources.assert_called_once_with()

    mock_config_changed.assert_called_once_with()
    set_state.assert_called_once_with("containerd.nvidia.ready")


@mock.patch.object(containerd, "application_version_set")
@mock.patch.object(containerd, "check_output")
def test_publish_version_to_juju(check_output, mock_version_set):
    """Verify containerd version parser."""
    check_output.return_value = b"containerd github.com/containerd/containerd 1.7.36"
    containerd.publish_version_to_juju()

    check_output.assert_called_once_with(["containerd", "--version"])
    mock_version_set.assert_called_once_with("1.7.36")


@mock.patch.object(containerd, "set_state")
@mock.patch.object(containerd, "remove_state")
@mock.patch.object(containerd, "is_state")
@mock.patch.object(containerd, "check_output")
@pytest.mark.parametrize(
    "params",
    [
        (False, None),
        (True, None),
        (True, CalledProcessError(-1, "nvidia-smi", output=b"just a fatal error")),
        (True, FileNotFoundError),
    ],
    ids=[
        "nvidia not available",
        "nvidia-smi returns without exception",
        "nvidia-smi returns with CalledProcessError (non-reboot exception)",
        "nvidia-smi returns with FileNotFound",
    ],
)
def test_needs_gpu_reboot_false(check_output, is_state, remove_state, set_state, params):
    """Verify situations where no gpu induced reboot is needed."""
    nvidia_available, nvidia_smi_exception = params
    is_state.return_value = nvidia_available
    check_output.side_effect = nvidia_smi_exception

    assert not containerd._test_gpu_reboot()
    if not nvidia_available:
        check_output.assert_not_called()
    else:
        check_output.assert_called_once_with(["nvidia-smi"], stderr=STDOUT)
    set_state.assert_not_called()
    remove_state.assert_called_once_with("containerd.nvidia.needs_reboot")


@mock.patch.object(containerd, "set_state")
@mock.patch.object(containerd, "remove_state")
@mock.patch.object(containerd, "is_state")
@mock.patch.object(containerd, "check_output")
def test_needs_gpu_reboot_true(check_output, is_state, remove_state, set_state):
    """Verify situations where a gpu induced reboot is needed."""
    is_state.return_value = True
    check_output.side_effect = CalledProcessError(-1, "nvidia-smi", output=b"Driver/library version mismatch")
    assert containerd._test_gpu_reboot()
    check_output.assert_called_once_with(["nvidia-smi"], stderr=STDOUT)
    set_state.assert_called_once_with("containerd.nvidia.needs_reboot")
    remove_state.assert_not_called()
