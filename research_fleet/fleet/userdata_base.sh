#!/usr/bin/env bash
# Thin EC2 userdata: install the BASE stack only (docker, node 22, python,
# build tools). The agent package, shim, secrets, and NemoClaw install are
# pushed over SSH afterward by provision_node.sh so we can watch/debug the
# critical NemoClaw sandbox build. Keeps userdata well under the 16KB limit.
set -eux
exec > /var/log/base-bootstrap.log 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ca-certificates curl gnupg git python3 python3-venv python3-pip \
  binutils zstd jq tmux unzip

install -m0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
. /etc/os-release
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" > /etc/apt/sources.list.d/docker.list
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
usermod -aG docker ubuntu || true

curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs

mkdir -p /opt/research-agent
chown -R ubuntu:ubuntu /opt/research-agent
touch /var/log/base-bootstrap.done
echo "base bootstrap complete"
