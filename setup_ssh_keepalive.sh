#!/bin/bash
# Run this on your GPU instance to fix SSH timeouts and install mosh

# 1. Set server-side SSH keepalive
sed -i 's/^#*ClientAliveInterval.*/ClientAliveInterval 60/' /etc/ssh/sshd_config
sed -i 's/^#*ClientAliveCountMax.*/ClientAliveCountMax 3/' /etc/ssh/sshd_config

# Add if not present
grep -q "ClientAliveInterval" /etc/ssh/sshd_config || echo "ClientAliveInterval 60" >> /etc/ssh/sshd_config
grep -q "ClientAliveCountMax" /etc/ssh/sshd_config || echo "ClientAliveCountMax 3" >> /etc/ssh/sshd_config

# Restart sshd
systemctl restart sshd 2>/dev/null || service ssh restart

# 2. Install mosh
apt-get update && apt-get install -y mosh

echo "Done. Now connect from your Mac using:"
echo "  mosh --ssh=\"ssh -p <PORT>\" root@<HOST>"
