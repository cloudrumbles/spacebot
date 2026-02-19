#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root (sudo)."
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  if [[ ! -f /etc/apt/keyrings/docker.asc ]]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
  fi

  source /etc/os-release
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list

  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

install -d -m 0755 /opt/spacebot
install -d -m 0750 /etc/spacebot
install -d -m 0755 /var/lib/spacebot

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cp "${script_dir}/docker-compose.yml" /opt/spacebot/docker-compose.yml
if [[ ! -f /etc/spacebot/spacebot.env ]]; then
  cp "${script_dir}/spacebot.env.example" /etc/spacebot/spacebot.env
  chmod 0640 /etc/spacebot/spacebot.env
  echo "Created /etc/spacebot/spacebot.env from template. Edit it before first start."
fi

cp "${script_dir}/spacebot-docker.service" /etc/systemd/system/spacebot.service
systemctl daemon-reload
systemctl enable --now spacebot.service

echo "Spacebot service installed."
echo "Next: edit /etc/spacebot/spacebot.env, then run: systemctl restart spacebot"
