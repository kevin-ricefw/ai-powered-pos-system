#!/bin/bash
# Run on the VM: bash ~/redeploy.sh
# Preserves whatever mounts (e.g. TLS certs) and port bindings the currently
# running veggielens-api container has, then swaps in the new image.
set -e

MOUNTS=$(sudo docker inspect veggielens-api --format '{{range .Mounts}}-v {{.Source}}:{{.Destination}}:{{if .RW}}rw{{else}}ro{{end}} {{end}}')
PORTS=$(sudo docker inspect veggielens-api --format '{{range $p, $conf := .HostConfig.PortBindings}}{{range $conf}}-p {{.HostPort}}:{{$p}} {{end}}{{end}}')
echo "Preserving mounts: $MOUNTS"
echo "Preserving ports:  $PORTS"

sudo docker stop veggielens-api
sudo docker rm veggielens-api
sudo docker load -i ~/veggielens-api.tar
sudo docker run -d --name veggielens-api --restart unless-stopped --env-file ~/.env $MOUNTS $PORTS veggielens-api:latest
