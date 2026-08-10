#!/usr/bin/env bash
# Launch the research-agent EC2 fleet in the GPU box's VPC/subnet.
# Boots N t3.xlarge Ubuntu 24.04 nodes with the thin base userdata; prints each
# instance id + public IP for provision_node.sh to finish over SSH.
#
# Usage: launch_fleet.sh <count>   (default 1 — launch node 1 first to validate)
set -euo pipefail
export AWS_SHARED_CREDENTIALS_FILE=/Users/billscolinos/.aws/credentials_cf AWS_PROFILE=cf_gpu AWS_DEFAULT_REGION=us-east-1
PREP=/Users/billscolinos/Documents/code_factory/staging/portfolio-agents-prep
COUNT="${1:-1}"

AMI=ami-052355af2a014bd2c          # Ubuntu 24.04
TYPE=t3.xlarge                     # 4 vCPU / 16 GB (NemoClaw+OpenShell needs >=8GB)
SUBNET=subnet-2e70a84a             # same subnet as the GPU box
SG=sg-0095be14b8bf0ed08            # research-fleet-sg
KEY=research-fleet-key

# focus areas: one per node (queue is shared; focus just biases task claim order)
FOCI=(momentum mean_reversion vol_target factor regime)

USERDATA_B64=$(base64 < "$PREP/fleet/userdata_base.sh")

for i in $(seq 1 "$COUNT"); do
  idx=$((i-1))
  AGENT_ID=$(printf "research-%02d" "$i")
  FOCUS="${FOCI[$idx]:-}"
  echo "== launching $AGENT_ID (focus=$FOCUS) =="
  IID=$(aws ec2 run-instances \
    --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
    --security-group-ids "$SG" --subnet-id "$SUBNET" --associate-public-ip-address \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=40,VolumeType=gp3}' \
    --instance-initiated-shutdown-behavior stop \
    --user-data "$USERDATA_B64" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$AGENT_ID},{Key=fleet,Value=research-agents},{Key=focus,Value=$FOCUS}]" \
    --query 'Instances[0].InstanceId' --output text)
  echo "  instance_id=$IID"
  echo "$AGENT_ID $IID $FOCUS" >> "$PREP/fleet/fleet_nodes.txt"
done

echo "== waiting for running + public IPs =="
aws ec2 wait instance-running --filters Name=tag:fleet,Values=research-agents Name=instance-state-name,Values=running 2>/dev/null || true
aws ec2 describe-instances --filters Name=tag:fleet,Values=research-agents Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].[Tags[?Key==`Name`].Value|[0],InstanceId,PublicIpAddress,PrivateIpAddress]' --output text
