#! usr/bin/bash
# This is a template script for authenticating aws account

CRED="credentials"

cat <<EOM >$CRED
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
EOM

mkdir ~/.aws
mv credentials ~/.aws/
CONFIG="config"
cat <<EOM >$CONFIG
[default]
region=eu-west-1
EOM
mv config ~/.aws/