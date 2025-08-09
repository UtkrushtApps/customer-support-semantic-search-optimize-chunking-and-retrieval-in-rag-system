#!/bin/bash
set -e
sudo docker-compose up -d chromadb
sleep 6
sudo docker-compose run --rm app python init_vector_db.py
echo "[Init] Support document vector DB ready."
