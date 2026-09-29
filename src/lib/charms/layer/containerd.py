import traceback

from charmhelpers.core import hookenv, unitdata
from charmhelpers.core.hookenv import log
from subprocess import check_output, CalledProcessError


def can_mount_cgroup2() -> bool:
    """Determine if it's possible to mount cgroup2 type filesystems."""
    try:
        stdout = check_output(["mount", "-t", "cgroup2"], text=True)
    except CalledProcessError:
        msg = "Failed to find mount type cgroup2\n" + traceback.format_exc()
        log(msg, level=hookenv.ERROR)
        return False
    return "type cgroup2" in stdout


def get_sandbox_image():
    """
    Return the container image location for the sandbox_image.

    Set an appropriate sandbox image based on known registries. Precedence should be:
    - related docker-registry
    - default charmed k8s registry (if related to kubernetes)
    - upstream

    :return: str container image location
    """
    db = unitdata.kv()
    canonical_registry = "ghcr.io/canonical/cdk"
    upstream_registry = "k8s.gcr.io"

    docker_registry = db.get("registry", None)
    if docker_registry:
        sandbox_registry = docker_registry["url"]
    else:
        try:
            deployment = hookenv.goal_state()
        except NotImplementedError:
            relations = []
            for rid in hookenv.relation_ids("containerd"):
                relations.append(hookenv.remote_service_name(rid))
        else:
            relations = deployment.get("relations", {}).get("containerd", {})

        if any(
            k in relations
            for k in (
                "kubernetes-control-plane",
                "kubernetes-master",  # wokeignore:rule=master
                "kubernetes-worker",
            )
        ):
            sandbox_registry = canonical_registry
        else:
            sandbox_registry = upstream_registry

    return "{}/pause:3.6".format(sandbox_registry)
