#!/bin/bash
cd pokemon-showdown
# --no-security disables authentication and rate limiting for local AI training
nohup node pokemon-showdown start --no-security > ../showdown_server.log 2>&1 &
echo $! > ../server.pid
echo "Server started with PID $(cat ../server.pid) (no-security mode)"
