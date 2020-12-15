#!/usr/bin/env bash

sudo cp service/ml-vis.service /etc/systemd/system/ml-vis.service
sudo systemctl daemon-reload
sudo systemctl enable ml-vis.service
