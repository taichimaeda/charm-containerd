import contextlib
import dataclasses
import os
import base64
import binascii
import json
from pathlib import Path
import re
import traceback
import typing

from subprocess import check_call, check_output, CalledProcessError, STDOUT
import urllib.request
import urllib.error

from charms.reactive import (
    hook,
    when,
    when_not,
    set_state,
    is_state,
    remove_state,
    endpoint_from_flag,
    register_trigger,
)

from charms.layer import containerd, status
from charms.layer.container_runtime_common import (
    ca_crt_path,
    server_crt_path,
    server_key_path,
    check_for_juju_https_proxy,
)

from charmhelpers.core import host, unitdata

from charmhelpers.core.templating import render
from charmhelpers.core.hookenv import atexit, config, env_proxy_settings, log, application_version_set

from charmhelpers.core.kernel import modprobe

from charmhelpers.fetch import (
    apt_cache,
    apt_install,
    apt_update,
    apt_purge,
    apt_hold,
    apt_autoremove,
    apt_unhold,
    import_key,
)

from charmhelpers.fetch.ubuntu_apt_pkg import Package

NVIDIA_SOURCES_FILE = "/etc/apt/sources.list.d/nvidia.list"
LEGACY_RESOURCE_BINARIES = {
    "containerd",
    "containerd-shim",
    "containerd-shim-runc-v1",
    "containerd-shim-runc-v2",
    "containerd-stress",
    "ctr",
}


def _containerd_version(value: str) -> typing.Tuple[int, int, int]:
    """Parse a containerd version."""
    match = re.search(r"(?<!\d)v?(\d+)\.(\d+)\.(\d+)(?!\d)", value)
    if not match:
        raise ValueError("Unable to determine containerd version from {!r}".format(value))
    return tuple(int(part) for part in match.groups())


def _containerd_config_version(version: typing.Tuple[int, int, int]) -> int:
    """Return the config version for a containerd version."""
    # NOTE: config_version is ignored. Containerd 1.x uses v2; 2.x uses v3.
    major = version[0]
    if major == 1:
        return 2
    if major == 2:
        return 3
    raise ValueError("Containerd {} is not supported".format(".".join(map(str, version))))


def _installed_containerd_version() -> typing.Tuple[int, int, int]:
    """Return the installed containerd version."""
    output = check_output(["containerd", "--version"]).decode()
    return _containerd_version(output)


# NOTE: This is public only because upgrade-actions.py imports it.
def candidate_containerd_version() -> typing.Tuple[int, int, int]:
    """Return the containerd version offered by apt."""
    package = apt_packages({CONTAINERD_PACKAGE}).get(CONTAINERD_PACKAGE)
    if package is None:
        raise RuntimeError("Containerd package is not available from apt")
    return _containerd_version(str(package.version))


def apt_packages(packages: typing.Set[str]) -> typing.Mapping[str, Package]:
    """Return a mapping of package names to Package classes.

    Ignores all packages which aren't available to apt
    Includes any package which wildcard matches with an installed package

    @param packages: List of packages for which to search
    @returns: Map of available packages
    """
    result = {}
    if not packages:
        return result

    cache = apt_cache()
    # also search for any already installed packages matching
    wildcards = [_ + "*" for _ in packages]
    packages = set(cache.dpkg_list(wildcards).keys()) | set(packages)

    for pkg_name in packages:
        try:
            result[pkg_name] = cache[pkg_name]
        except KeyError:
            log(f"Cannot find {pkg_name} in apt.")
    return result


@contextlib.contextmanager
def proxy_env():
    """Create a context to temporarily modify proxy in os.environ."""
    restore = {**os.environ}  # Copy the current os.environ
    # overwrite JUJU_CHARM_*_PROXY from config where available
    for key in ["http_proxy", "https_proxy", "no_proxy"]:
        val = config(key)
        if val:
            os.environ[f"JUJU_CHARM_{key.upper()}"] = val
    juju_proxies = env_proxy_settings() or {}
    os.environ.update(**juju_proxies)  # Insert or Update the os.environ
    yield os.environ
    for key in juju_proxies:
        del os.environ[key]  # remove any keys which were added or updated
    os.environ.update(**restore)  # restore any original values


def fetch_url_text(urls) -> typing.List[typing.Optional[str]]:
    """Fetch url text within a proxied environment.

    returns: None in the event the url yielded no response.
    """
    # updates os.environ to include juju proxy settings
    with proxy_env():
        handlers = [urllib.request.ProxyHandler()]
        opener = urllib.request.build_opener(*handlers)
        responses = []
        for url in urls:
            resp = None
            try:
                resp = opener.open(url).read().decode()
            except urllib.error.HTTPError as e:
                log(f"Cannot fetch url='{url}' with code {e.code} {e.reason}")
            except urllib.error.URLError as e:
                log(f"Cannot fetch url='{url}' with {e.reason}")
            finally:
                responses.append(resp)

    return responses


DB = unitdata.kv()

CONTAINERD_PACKAGE = "containerd"

register_trigger(when="config.changed.nvidia_apt_sources", clear_flag="containerd.nvidia.ready")
register_trigger(when="config.changed.nvidia_apt_packages", clear_flag="containerd.nvidia.ready")


def _check_containerd():
    """Check containerd and its CRI plugins."""
    # NOTE: ctr version checks connectivity only. CRI plugins are checked below.
    try:
        version = _installed_containerd_version()
        config_version = _containerd_config_version(version)

        # ctr version checks that the client can reach the containerd server.
        check_output(["ctr", "version"])
        # ctr plugins ls checks that the CRI plugins are healthy.
        plugins = check_output(["ctr", "plugins", "ls"]).decode()
    except (FileNotFoundError, CalledProcessError, ValueError):
        return False

    expected_plugins = {
        2: {
            ("io.containerd.grpc.v1", "cri"),
        },
        3: {
            ("io.containerd.grpc.v1", "cri"),
            ("io.containerd.cri.v1", "images"),
            ("io.containerd.cri.v1", "runtime"),
        },
    }[config_version]

    healthy_plugins = set()
    for line in plugins.splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue

        plugin_type, plugin_id, _, status = fields
        if status == "ok":
            healthy_plugins.add((plugin_type, plugin_id))

    missing_plugins = expected_plugins - healthy_plugins
    if missing_plugins:
        missing = ", ".join("{}/{}".format(*plugin) for plugin in sorted(missing_plugins))
        log("Containerd CRI plugins are not healthy: {}".format(missing))
        return False

    return True


def _juju_proxy_changed():
    """
    Check to see if the Juju model HTTP(S) proxy settings have changed.

    These aren't propagated to the charm so we'll need to do it here.

    :return: Boolean
    """
    cached = DB.get("config-cache", None)
    if not cached:
        return True  # First pass.

    new = check_for_juju_https_proxy(config)

    if (
        cached["http_proxy"] == new["http_proxy"]
        and cached["https_proxy"] == new["https_proxy"]
        and cached["no_proxy"] == new["no_proxy"]
    ):
        return False

    return True


@when("containerd.nvidia.needs_reboot")
def _test_gpu_reboot() -> bool:
    reboot = False
    if is_state("containerd.nvidia.available"):
        try:
            check_output(["nvidia-smi"], stderr=STDOUT)
        except CalledProcessError as cpe:
            log("Unable to communicate with the NVIDIA driver.")
            log(cpe)
            reboot = any(message in cpe.stdout.decode() for message in ["Driver/library version mismatch"])
        except FileNotFoundError as fne:
            log("NVIDIA SMI not installed.")
            log(fne)
    if reboot:
        set_state("containerd.nvidia.needs_reboot")
    else:
        remove_state("containerd.nvidia.needs_reboot")
    return reboot


@atexit
def charm_status():
    """
    Set the charm's status after each hook is run.

    :return: None
    """
    if is_state("upgrade.series.in-progress"):
        status.blocked("Series upgrade in progress")
    elif is_state("containerd.resource-migration.failed"):
        status.blocked("Failed to migrate containerd binaries to apt")
    elif is_state("containerd.install.failed"):
        status.blocked("Failed to install containerd from apt")
    elif is_state("containerd.custom-registries.invalid"):
        status.blocked("Invalid custom_registries configuration")
    elif is_state("containerd.registry-render.failed"):
        status.blocked("Failed to render registry configuration")
    elif is_state("containerd.config-render.failed"):
        status.blocked("Failed to render containerd configuration")
    elif is_state("containerd.nvidia.invalid-option"):
        status.blocked("{} is an invalid option for gpu_driver".format(config().get("gpu_driver")))
    elif is_state("containerd.nvidia.fetch_keys_failed"):
        status.blocked("Failed to fetch nvidia_apt_key_urls.")
    elif is_state("containerd.nvidia.missing_package_list"):
        status.blocked("No NVIDIA packages selected to install.")
    elif is_state("containerd.nvidia.needs_reboot"):
        status.blocked("May need reboot to activate GPU.")
    # NOTE: Keep the pre-existing restart retry flow waiting until it succeeds.
    elif is_state("containerd.restart"):
        status.waiting("Containerd restart pending")
    # TODO: Handle other logged/retried failures that can leave the unit falsely active/blocked.
    elif _check_containerd():
        status.active("Container runtime available")
        set_state("containerd.ready")
    else:
        status.blocked("Container runtime not available")


def strip_url(url):
    """Strip the URL of protocol, slashes etc., and keep host:port.

    Examples:
        url: http://10.10.10.10:8000 --> return: 10.10.10.10:8000
        url: https://myregistry.io:8000/ --> return: myregistry.io:8000
        url: myregistry.io:8000 --> return: myregistry.io:8000
    """
    return url.rstrip("/").split(sep="://", maxsplit=1)[-1]


def update_custom_tls_config(config_directory, registries, old_registries):
    """
    Read registries config and remove old/write new tls files from/to disk.

    :param str config_directory: containerd config directory
    :param List registries: juju config for custom registries
    :param List old_registries: old juju config for custom registries
    :return: None
    """
    # Remove tls files of old registries; so not to leave uneeded, stale files.
    for registry in old_registries:
        registry.uninstall_tls(config_directory)

    # Write tls files of new registries.
    for registry in registries:
        registry.install_tls(config_directory)


def insert_docker_io_to_custom_registries(custom_registries):
    """
    Ensure the default docker.io registry exists.

    Also gives a way for configuration to override the url for it.
    If a docker.io host entry doesn't exist, we'll add one.
    """
    if isinstance(custom_registries, list):
        if not any(d.host == "docker.io" for d in custom_registries):
            custom_registries.insert(0, Registry(url="https://registry-1.docker.io", host="docker.io"))
    return custom_registries


class ValidationError(Exception):
    """Defines an error for an invalid custom registry."""


class DuplicateError(ValidationError):
    """Defines an error for duplicate hosts in a custom registry."""

    code = "duplicate"
    msg_template = "host defines {host} more than once at {idx}"


@dataclasses.dataclass
class Registry:
    """Define the structure of a custom registry."""

    url: str
    host: typing.Union[str, None] = None
    username: typing.Union[str, None] = None
    password: typing.Union[str, dict, None] = None
    ca_file: typing.Union[str, None] = None
    cert_file: typing.Union[str, None] = None
    key_file: typing.Union[str, None] = None
    insecure_skip_verify: typing.Union[bool, None] = None

    ca: typing.Union[str, None] = dataclasses.field(init=False, default=None)
    cert: typing.Union[str, None] = dataclasses.field(init=False, default=None)
    key: typing.Union[str, None] = dataclasses.field(init=False, default=None)

    def __post_init__(self):
        """Populate host field from url if missing.

        Examples:
            url: http://10.10.10.10:8000 --> host: 10.10.10.10:8000
            url: https://myregistry.io:8000/ --> host: myregistry.io:8000
            url: myregistry.io:8000 --> host: myregistry.io:8000
        """
        if self.host is None:
            self.host = strip_url(self.url)
        # NOTE: Keep registry hosts inside certs.d.
        if not self.host or self.host in (".", "..") or "/" in self.host:
            raise ValidationError("registry host {!r} is not valid".format(self.host))

    # NOTE: config.toml auth keys and hosts.toml endpoints use different URL forms.
    @property
    def server(self) -> str:
        """Return the fallback registry."""
        if self.host == strip_url(self.url):
            return self.url_normalized
        return "https://{}".format(self.host)

    @property
    def url_stripped(self) -> str:
        """Return the URL without its scheme."""
        return strip_url(self.url)

    @property
    def url_normalized(self) -> str:
        """Return the URL with a scheme and no trailing slash."""
        url = self.url.rstrip("/")
        if url.startswith("http://"):
            return url
        if url.startswith("https://"):
            return url
        return "https://{}".format(url)

    @classmethod
    def from_dict(cls, idx: int, value: typing.Mapping[str, typing.Any]):
        """Build a Registry object from a dict."""
        for field in dataclasses.fields(cls):
            field_type = typing.get_origin(field.type)
            field_args = typing.get_args(field.type)
            field_value = value.get(field.name)
            optional = field_type is typing.Union and type(None) in field_args
            if not optional and field.name not in value:
                raise ValidationError("registry #{} missing required field '{}'".format(idx, field.name))
            allowed_types = field_args or (field.type,)
            if not isinstance(field_value, allowed_types):
                allowed_types = [_.__name__ for _ in allowed_types]
                type_hint = ",".join([_ for _ in allowed_types if _ != "NoneType"])
                raise ValidationError(
                    "registry #{} field {}={} is type {}, not type {}".format(
                        idx, field.name, field_value, type(field_value).__name__, type_hint
                    )
                )
        allowed_field_names = {_.name for _ in dataclasses.fields(cls)}
        for field in value.keys() - allowed_field_names:
            raise ValidationError("registry #{} field {} may not be specified".format(idx, field))
        return cls(**value)

    def _write_tls_content(self, content: str, opt: str, config_directory: str) -> typing.Optional[str]:
        if not content:
            return None
        try:
            file_contents = base64.b64decode(content)
        except (binascii.Error, TypeError):
            log(traceback.format_exc())
            log("{}:{} didn't look like base64 data... skipping".format(self.url, opt))
            return None
        file_path = os.path.join(config_directory, "%s.%s" % (self.host, opt))
        with open(file_path, "wb") as f:
            f.write(file_contents)
        # NOTE: Client keys are root-only.
        os.chmod(file_path, 0o600 if opt == "key" else 0o644)
        return file_path

    def _remove_tls_content(self, opt: str, config_directory: str) -> None:
        file_path = os.path.join(config_directory, "%s.%s" % (self.host, opt))
        if os.path.isfile(file_path):
            os.remove(file_path)

    def install_tls(self, config_directory):
        """Install tls content onto the file system."""
        self.ca = self._write_tls_content(self.ca_file, "ca", config_directory)
        self.key = self._write_tls_content(self.key_file, "key", config_directory)
        self.cert = self._write_tls_content(self.cert_file, "cert", config_directory)

    def uninstall_tls(self, config_directory):
        """Remove tls content from the file system."""
        self.ca = self._remove_tls_content("ca", config_directory)
        self.key = self._remove_tls_content("key", config_directory)
        self.cert = self._remove_tls_content("cert", config_directory)


@dataclasses.dataclass
class RegistryList:
    """Definition for a Json String representing a list of custom registries."""

    registries: typing.List[Registry]

    @classmethod
    def from_json(cls, value: str):
        """Build a registry list from json."""
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            raise ValidationError("Failed to decode json string")
        if not isinstance(parsed, list):
            raise ValidationError("custom_registries is not a list")

        full_list, host_set = [], set()
        for idx, registry in enumerate(parsed):
            if not isinstance(registry, dict):
                raise ValidationError("registry #{} is not in object form".format(idx))
            registry = Registry.from_dict(idx, registry)

            if registry.host in host_set:
                raise DuplicateError("registry #{} defines {} more than once".format(idx, registry.host))
            host_set.add(registry.host)
            full_list.append(registry)
        return cls(full_list)


def _registries_list(registries: str, default=None):
    """
    Parse registry config and ensure it returns a list or raises ValueError.

    :param str registries: representation of registries
    :param default: if provided, return rather than raising exceptions
    :return: List of registry objects
    """
    validated = default
    try:
        validated = [r for r in RegistryList.from_json(registries).registries]
    except ValidationError:
        if default is None:
            raise
    return validated


def _insert_registry_from_relation(registries):
    """Add the relation registry."""
    db_registry = DB.get("registry", None)
    if not db_registry:
        return registries
    db_host = db_registry.get("host") or strip_url(db_registry["url"])

    # NOTE: Relation and custom registries cannot use the same host.
    hosts = {registry.host for registry in registries}
    if db_host in hosts:
        raise DuplicateError("Duplicate registry host configuration {}".format(db_host))

    registry = Registry(
        url=db_registry["url"],
        host=db_host,
        username=db_registry.get("username"),
        password=db_registry.get("password"),
        ca_file=db_registry.get("ca_file"),
        cert_file=db_registry.get("cert_file"),
        key_file=db_registry.get("key_file"),
        insecure_skip_verify=db_registry.get("insecure_skip_verify"),
    )
    registry.ca = db_registry.get("ca")
    registry.cert = db_registry.get("cert")
    registry.key = db_registry.get("key")
    registries.append(registry)

    return registries


def merge_custom_registries(config_directory, custom_registries, old_custom_registries):
    """
    Merge custom registries and Docker registries from relation.

    :param str config_directory: containerd config directory
    :param str custom_registries: juju config for custom registries
    :param str old_custom_registries: old juju config for custom registries
    :return: List Dictionary merged registries
    """
    registries = _registries_list(custom_registries, default=[])
    old_registries = []
    if old_custom_registries:
        old_registries += _registries_list(old_custom_registries, default=[])
    update_custom_tls_config(config_directory, registries, old_registries)

    registries = _insert_registry_from_relation(registries)
    registries = insert_docker_io_to_custom_registries(registries)

    return registries


def invalid_custom_registries(custom_registries):
    """
    Validate custom registries from config.

    :param str custom_registries: juju config for custom registries
    :return: error string for blocked status if condition exists, None otherwise
    :rtype: Optional[str]
    """
    try:
        _registries_list(custom_registries)
    except ValidationError as e:
        return str(e)


@hook("update-status")
def update_status():
    """
    Triggered when update-status is called.

    :return: None
    """
    if _juju_proxy_changed():
        set_state("containerd.juju-proxy.changed")


@hook("upgrade-charm")
def upgrade_charm():
    """
    Triggered when upgrade-charm is called.

    :return: None
    """
    # Prevent containerd apt pkg from being implicitly updated.
    apt_hold(CONTAINERD_PACKAGE)

    if not is_state("containerd.resource.installed"):
        # Apply configuration template changes shipped by the new charm.
        # Resource migration renders candidate-compatible config itself.
        config_changed()

    # Clean up old nvidia sources.list.d files
    old_source_files = [
        "/etc/apt/sources.list.d/nvidia-container-runtime.list",
        "/etc/apt/sources.list.d/cuda.list",
    ]
    for source_file in old_source_files:
        if os.path.exists(source_file):
            os.remove(source_file)
            remove_state("containerd.nvidia.ready")

    # Update containerd version
    remove_state("containerd.version-published")


# NOTE: Keep this handler separate so failed migrations retry on later hooks.
@when("containerd.resource.installed")
@when_not("endpoint.containerd.departed")
def migrate_resource_containerd():
    """Replace resource binaries with the apt package."""
    status.maintenance("Migrating containerd binaries to apt")
    try:
        apt_update(fatal=True)
        resource_version = _installed_containerd_version()
        candidate_version = candidate_containerd_version()
        if candidate_version < resource_version:
            raise RuntimeError(
                "apt candidate {} is older than resource containerd {}".format(
                    ".".join(map(str, candidate_version)),
                    ".".join(map(str, resource_version)),
                )
            )

        reinstall_containerd(candidate_version)

        # Remove resource binaries not owned by the apt package.
        for binary in LEGACY_RESOURCE_BINARIES:
            path = Path("/usr/bin") / binary
            if not os.path.lexists(path):
                continue
            try:
                check_output(["dpkg-query", "--search", path], stderr=STDOUT)
            except CalledProcessError:
                log("Removing unowned legacy resource binary {}".format(path))
                os.remove(path)
    except (CalledProcessError, OSError, RuntimeError, ValueError):
        log(traceback.format_exc())
        set_state("containerd.resource-migration.failed")
        return
    remove_state("containerd.resource-migration.failed")

    # Clear legacy flags for resource binaries.
    remove_state("containerd.resource.installed")
    remove_state("containerd.resource.evaluated")

    set_state("containerd.installed")


@when_not("containerd.br_netfilter.enabled")
def enable_br_netfilter_module():
    """
    Enable br_netfilter to work around https://github.com/kubernetes/kubernetes/issues/21613.

    :return: None
    """
    try:
        modprobe("br_netfilter", persist=True)
    except Exception:
        log(traceback.format_exc())
        if host.is_container():
            log("LXD detected, ignoring failure to load br_netfilter")
        else:
            log("LXD not detected, will retry loading br_netfilter")
            return
    set_state("containerd.br_netfilter.enabled")


@when_not("containerd.ready", "containerd.installed", "endpoint.containerd.departed")
def install_containerd():
    """
    Install containerd and then create initial configuration.

    :return: None
    """
    status.maintenance("Installing containerd via apt")
    try:
        apt_update(fatal=True)
        candidate_version = candidate_containerd_version()
        reinstall_containerd(candidate_version)
        set_state("containerd.installed")
        remove_state("containerd.install.failed")
    except (CalledProcessError, OSError, RuntimeError, ValueError):
        log(traceback.format_exc())
        set_state("containerd.install.failed")


def reinstall_containerd(candidate_version) -> None:
    """Render config and reinstall containerd from apt."""
    # Render first because apt may restart the newly installed binary.
    if not _render_config(version=candidate_version):
        raise RuntimeError("failed to render configuration for apt candidate")

    apt_unhold(CONTAINERD_PACKAGE)
    try:
        apt_install([CONTAINERD_PACKAGE, "--reinstall"], fatal=True)
    except CalledProcessError:
        try:
            remaining_version = _installed_containerd_version()
        except (CalledProcessError, FileNotFoundError, ValueError):
            raise
        if not _render_config(version=remaining_version):
            raise RuntimeError("failed to restore configuration after apt failure")
        if not host.service_restart("containerd.service"):
            raise RuntimeError("failed to restart containerd after apt failure")
        if not _check_containerd():
            raise RuntimeError("containerd CRI plugins are not healthy after apt failure")
        raise
    finally:
        apt_hold(CONTAINERD_PACKAGE)

    if not host.service_restart("containerd.service"):
        raise RuntimeError("failed to restart containerd after apt reinstall")
    if not _check_containerd():
        raise RuntimeError("containerd CRI plugins are not healthy after apt reinstall")


@when("containerd.installed")
@when_not("containerd.version-published")
def publish_version_to_juju():
    """
    Publish the containerd version to Juju.

    :return: None
    """
    try:
        version = _installed_containerd_version()
    except (FileNotFoundError, CalledProcessError, ValueError):
        return

    application_version_set(".".join(map(str, version)))
    set_state("containerd.version-published")


@when_not("containerd.nvidia.checked")
@when_not("endpoint.containerd.departed")
def check_for_gpu():
    """
    Check if an Nvidia GPU exists.

    :return: None
    """
    valid_options = ["auto", "none", "nvidia"]

    driver_config = config().get("gpu_driver")
    if driver_config not in valid_options:
        set_state("containerd.nvidia.invalid-option")
        return

    out = check_output(["lspci", "-nnk"]).rstrip().decode("utf-8").lower()
    nvidia_pci, auto = out.count("nvidia"), driver_config == "auto"

    if driver_config == "none" or (auto and not nvidia_pci):
        # prevent/remove nvidia driver from activating
        # because of config or no nvidia hardware found
        remove_state("containerd.nvidia.available")

    if driver_config == "nvidia" or (auto and nvidia_pci):
        # allow/install nvidia drivers to activate
        # because of config or this found nvidia hardware
        set_state("containerd.nvidia.available")

    remove_state("containerd.nvidia.invalid-option")
    set_state("containerd.nvidia.checked")


def _configured_nvidia_packages():
    # Workaround for LP#2017175 where cuda-drivers end up depending on
    # screen-resolution-extra which in focal needs either gnome-shell or policykit-1-gnome
    # By adding policykit-1-gnome, it fulfills the dependency and doesn't add gnome-shell

    # See also bug on screen-resolution-extra LP#1930937
    pkgs = set(config("nvidia_apt_packages").split())
    dist = host.lsb_release()
    if dist["DISTRIB_CODENAME"].lower() == "focal":
        pkgs.add("policykit-1-gnome")
    return list(pkgs)


@when("containerd.nvidia.ready")
@when_not("containerd.nvidia.available")
def unconfigure_nvidia(reconfigure=True):
    """
    Based on charm config, remove NVIDIA drivers.

    :return: None
    """
    status.maintenance("Removing NVIDIA drivers.")

    nvidia_packages = _configured_nvidia_packages()
    apt_unhold(nvidia_packages)
    to_purge = apt_packages(nvidia_packages).keys()

    if to_purge:
        # remove any other nvidia- installed packages
        apt_purge(to_purge | {"^nvidia-.*"}, fatal=True)

    if os.path.isfile(NVIDIA_SOURCES_FILE):
        os.remove(NVIDIA_SOURCES_FILE)

    if to_purge:
        apt_autoremove(purge=True, fatal=True)

    remove_state("containerd.nvidia.ready")
    if reconfigure:
        config_changed()


@when("containerd.nvidia.available", "config.changed.nvidia_apt_key_urls")
def configure_nvidia_sources():
    """Configure NVIDIA repositories based on charm config.

    :return: bool - True if successufully fetched
    """
    status.maintenance("Configuring NVIDIA repositories.")

    dist = host.lsb_release()
    os_release_id = dist["DISTRIB_ID"].lower()
    os_release_version_id = dist["DISTRIB_RELEASE"]
    os_release_version_id_no_dot = os_release_version_id.replace(".", "")

    key_urls = config("nvidia_apt_key_urls").split()
    formatted_key_urls = [
        key_url.format(
            id=os_release_id,
            version_id=os_release_version_id,
            version_id_no_dot=os_release_version_id_no_dot,
        )
        for key_url in key_urls
    ]
    if formatted_key_urls:
        fetched_keys = fetch_url_text(formatted_key_urls)
        if not all(fetched_keys):
            set_state("containerd.nvidia.fetch_keys_failed")
            return False
        remove_state("containerd.nvidia.fetch_keys_failed")

        for key in fetched_keys:
            import_key(key)

    if os.path.isfile(NVIDIA_SOURCES_FILE):
        os.remove(NVIDIA_SOURCES_FILE)

    sources = config("nvidia_apt_sources").splitlines()
    formatted_sources = [
        source.format(
            id=os_release_id,
            version_id=os_release_version_id,
            version_id_no_dot=os_release_version_id_no_dot,
        )
        for source in sources
    ]
    with open(NVIDIA_SOURCES_FILE, "w") as f:
        f.write("\n".join(formatted_sources))

    return True


@when("containerd.nvidia.available")
@when_not("containerd.nvidia.ready", "endpoint.containerd.departed")
def install_nvidia_drivers(reconfigure=True):
    """Based on charm config, install and configure NVIDIA drivers.

    :return: None
    """
    # Fist remove any existing nvidia drivers
    unconfigure_nvidia(reconfigure=False)
    if not configure_nvidia_sources():
        return

    status.maintenance("Installing NVIDIA drivers.")
    apt_update()
    nvidia_packages = _configured_nvidia_packages()
    if not nvidia_packages:
        set_state("containerd.nvidia.missing_package_list")
        return
    remove_state("containerd.nvidia.missing_package_list")

    options = [
        "--option=Dpkg::Options::=--force-confold",
        "--no-install-recommends",
    ]
    apt_install(nvidia_packages, fatal=True, options=options)
    # Prevent nvidia packages from being automatically updated.
    apt_hold(nvidia_packages)
    _test_gpu_reboot()

    set_state("containerd.nvidia.ready")
    if reconfigure:
        config_changed()


@when("endpoint.containerd.departed")
def purge_containerd():
    """
    Purge Containerd from the cluster.

    :return: None
    """
    status.maintenance("Removing containerd from principal")

    host.service_stop("containerd.service")
    apt_unhold(CONTAINERD_PACKAGE)
    apt_purge(CONTAINERD_PACKAGE, fatal=True)

    if is_state("containerd.nvidia.ready"):
        unconfigure_nvidia(reconfigure=False)

    apt_autoremove(purge=True, fatal=True)

    remove_state("containerd.ready")
    remove_state("containerd.installed")
    remove_state("containerd.nvidia.ready")
    remove_state("containerd.nvidia.checked")
    remove_state("containerd.nvidia.available")
    remove_state("containerd.version-published")


@when("config.changed.gpu_driver")
def gpu_config_changed():
    """
    Remove the GPU checked state when the config is changed.

    :return: None
    """
    remove_state("containerd.nvidia.checked")


CONFIG_DIRECTORY = "/etc/containerd"
CONFIG_FILE = "config.toml"
REGISTRY_CONFIG_DIRECTORY = "certs.d"
REGISTRY_CONFIG_FILE = "hosts.toml"


def _render_registry_config(config_directory: str, registries: typing.List[Registry]) -> None:
    """Render hosts.toml files and remove stale files."""
    registry_directory = Path(config_directory) / REGISTRY_CONFIG_DIRECTORY
    os.makedirs(registry_directory, mode=0o755, exist_ok=True)

    current_hosts = {registry.host for registry in registries}
    previous_hosts = set(DB.get("registry-hosts", []))

    for registry in registries:
        host_directory = registry_directory / registry.host
        os.makedirs(host_directory, mode=0o755, exist_ok=True)
        hosts_file = host_directory / REGISTRY_CONFIG_FILE
        # NOTE: Keep registry host config root-only.
        render(
            REGISTRY_CONFIG_FILE,
            str(hosts_file),
            {"registry": registry},
            perms=0o600,
        )

    for host_name in previous_hosts - current_hosts:
        host_directory = registry_directory / host_name
        hosts_file = host_directory / REGISTRY_CONFIG_FILE
        try:
            os.remove(hosts_file)
        except FileNotFoundError:
            pass
        try:
            os.rmdir(host_directory)
        except (FileNotFoundError, OSError):
            pass

    DB.set("registry-hosts", sorted(current_hosts))


# NOTE: Package transitions restart containerd directly after rendering.
def _render_config(version=None):
    """
    Render the config template.

    :param version: version whose native config schema should be rendered
    :return: whether configuration was rendered successfully
    :rtype: bool
    """
    if _juju_proxy_changed():
        set_state("containerd.juju-proxy.changed")

    # Create "dumb" context based on Config to avoid triggering config.changed
    context = dict(config())

    if not version:
        version = _installed_containerd_version()
    config_version = _containerd_config_version(version)
    template_config = "config_v{}.toml".format(config_version)

    # Configure runtime type
    context["runtime_type"] = "io.containerd.runc.v2"

    if config_version == 2 and not containerd.can_mount_cgroup2():
        context["runtime_type"] = "io.containerd.runc.v1"

    endpoint = endpoint_from_flag("endpoint.containerd.available")
    if endpoint:
        sandbox_image = endpoint.get_sandbox_image()
        if sandbox_image:
            log("Setting sandbox_image to: {}".format(sandbox_image))
            context["sandbox_image"] = sandbox_image
        else:
            context["sandbox_image"] = containerd.get_sandbox_image()
    else:
        context["sandbox_image"] = containerd.get_sandbox_image()

    if not os.path.isdir(CONFIG_DIRECTORY):
        os.mkdir(CONFIG_DIRECTORY)

    # If custom_registries changed, make sure to remove old tls files.
    if config().changed("custom_registries"):
        old_custom_registries = config().previous("custom_registries")
    else:
        old_custom_registries = None

    # validate custom_registries
    invalid_reason = invalid_custom_registries(context["custom_registries"])
    if invalid_reason:
        set_state("containerd.custom-registries.invalid")
        log(invalid_reason)
        return False
    remove_state("containerd.custom-registries.invalid")

    try:
        context["custom_registries"] = merge_custom_registries(
            CONFIG_DIRECTORY,
            context["custom_registries"],
            old_custom_registries,
        )
        if config_version == 3:
            _render_registry_config(CONFIG_DIRECTORY, context["custom_registries"])
    except (OSError, ValidationError):
        log(traceback.format_exc())
        set_state("containerd.registry-render.failed")
        return False
    remove_state("containerd.registry-render.failed")

    untrusted = DB.get("untrusted")
    if untrusted:
        context["untrusted"] = True
        context["untrusted_name"] = untrusted["name"]
        context["untrusted_path"] = untrusted["binary_path"]
        context["untrusted_binary"] = os.path.basename(untrusted["binary_path"])

    else:
        context["untrusted"] = False

    if context.get("runtime") == "auto":
        if is_state("containerd.nvidia.available"):
            context["runtime"] = "nvidia-container-runtime"
        else:
            context["runtime"] = "runc"

    try:
        # NOTE: Keep config.toml root-only because it can contain registry credentials.
        render(template_config, str(Path(CONFIG_DIRECTORY) / CONFIG_FILE), context, perms=0o600)
    except OSError:
        log(traceback.format_exc())
        set_state("containerd.config-render.failed")
        return False
    remove_state("containerd.config-render.failed")

    return True


@when("config.changed")
@when_not("endpoint.containerd.departed")
def config_changed():
    """Render config and request a restart."""
    if _render_config():
        set_state("containerd.restart")


@when("containerd.installed")
@when("config.changed.kill_signal")
@when_not("endpoint.containerd.departed")
def render_kill_signal():
    """
    Apply new kill-signal settings.

    :return: None
    """
    service_file = "containerd_kill.conf"
    service_directory = "/etc/systemd/system/containerd.service.d"
    service_path = os.path.join(service_directory, service_file)

    os.makedirs(service_directory, exist_ok=True)

    log("Applying kill signal, writing new file to {}".format(service_path))
    context = dict(kill_signal=config().get("kill_signal"))
    render(service_file, service_path, context)

    check_call(["systemctl", "daemon-reload"])
    set_state("containerd.restart")


@when("containerd.installed")
@when("containerd.juju-proxy.changed")
@when_not("endpoint.containerd.departed")
def proxy_changed():
    """
    Apply new proxy settings.

    :return: None
    """
    # Create "dumb" context based on Config
    # to avoid triggering config.changed.
    context = check_for_juju_https_proxy(config)

    service_file = "proxy.conf"
    service_directory = "/etc/systemd/system/containerd.service.d"
    service_path = os.path.join(service_directory, service_file)

    if context.get("http_proxy") or context.get("https_proxy") or context.get("no_proxy"):
        os.makedirs(service_directory, exist_ok=True)

        log("Proxy changed, writing new file to {}".format(service_path))
        render(service_file, service_path, context)

    else:
        try:
            log("Proxy cleaned, removing file {}".format(service_path))
            os.remove(service_path)
        except FileNotFoundError:
            return  # We don't need to restart the daemon.

    DB.set("config-cache", context)

    remove_state("containerd.juju-proxy.changed")
    check_call(["systemctl", "daemon-reload"])
    set_state("containerd.restart")


@when("containerd.restart")
@when_not("endpoint.containerd.departed")
def restart_containerd():
    """
    Restart the containerd service.

    If the restart fails, this function will log a message and be retried on
    the next hook.
    """
    status.maintenance("Restarting containerd")
    if not host.service_restart("containerd.service"):
        log("Failed to restart containerd; will retry")
        return
    if not _check_containerd():
        log("CRI plugins are not healthy; will retry")
        return
    remove_state("containerd.restart")


@when("containerd.ready")
@when("endpoint.containerd.joined")
@when_not("endpoint.containerd.departed")
def publish_config():
    """
    Pass configuration to principal charm.

    :return: None
    """
    endpoint = endpoint_from_flag("endpoint.containerd.joined")
    endpoint.set_config(
        socket="unix:///var/run/containerd/containerd.sock",
        runtime="remote",  # TODO handle in k8s worker.
        nvidia_enabled=is_state("containerd.nvidia.available"),
    )


@when("endpoint.untrusted.available")
@when_not("untrusted.configured")
@when_not("endpoint.containerd.departed")
def untrusted_available():
    """
    Handle untrusted container runtime.

    :return: None
    """
    untrusted_runtime = endpoint_from_flag("endpoint.untrusted.available")
    received = dict(untrusted_runtime.get_config())

    if "name" not in received.keys():
        return  # Try until config is available.

    DB.set("untrusted", received)
    config_changed()

    set_state("untrusted.configured")


@when("endpoint.untrusted.departed")
def untrusted_departed():
    """
    Handle untrusted container runtime.

    :return: None
    """
    DB.unset("untrusted")
    DB.flush()
    config_changed()

    remove_state("untrusted.configured")


@when("endpoint.docker-registry.ready")
@when_not("containerd.registry.configured")
def configure_registry():
    """
    Add docker registry config when present.

    :return: None
    """
    registry = endpoint_from_flag("endpoint.docker-registry.ready")

    docker_registry = Registry(url=registry.registry_netloc)

    # Handle auth data.
    if registry.has_auth_basic():
        docker_registry.username = registry.basic_user
        docker_registry.password = registry.basic_password

    # Handle TLS data.
    if registry.has_tls():
        # Ensure the CA that signed our registry cert is trusted.
        host.install_ca_cert(registry.tls_ca, name="juju-docker-registry")

        docker_registry.ca = str(ca_crt_path)
        docker_registry.key = str(server_key_path)
        docker_registry.cert = str(server_crt_path)

    DB.set("registry", dataclasses.asdict(docker_registry))

    config_changed()
    set_state("containerd.registry.configured")


@when("endpoint.docker-registry.changed", "containerd.registry.configured")
def reconfigure_registry():
    """
    Signal to update the registry config when something changes.

    :return: None
    """
    remove_state("containerd.registry.configured")


@when("endpoint.containerd.reconfigure")
@when_not("endpoint.containerd.departed")
def container_runtime_relation_changed():
    """
    Run config_changed to use any new config from the endpoint.

    :return: None
    """
    config_changed()
    endpoint = endpoint_from_flag("endpoint.containerd.reconfigure")
    endpoint.handle_remote_config()


@when("containerd.registry.configured")
@when_not("endpoint.docker-registry.joined")
def remove_registry():
    """
    Remove registry config when the registry is no longer present.

    :return: None
    """
    docker_registry = DB.get("registry", None)

    if docker_registry:
        # Remove from DB.
        DB.unset("registry")
        DB.flush()

        # Remove auth-related data.
        log("Disabling auth for docker registry: {}.".format(docker_registry["url"]))

    config_changed()
    remove_state("containerd.registry.configured")
