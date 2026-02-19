# systemd User Service

Install the service:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/spacebot.service ~/.config/systemd/user/spacebot.service
systemctl --user daemon-reload
systemctl --user enable --now spacebot.service
```

Verify:

```bash
systemctl --user status spacebot.service
journalctl --user -u spacebot.service -f
```

Optional (keep user service running after logout):

```bash
loginctl enable-linger "$USER"
```
