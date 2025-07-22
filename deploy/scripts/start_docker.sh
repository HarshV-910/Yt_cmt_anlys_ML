#!/bin/bash
# Login to AWS ECR
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging into ECR..."
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 252312374343.dkr.ecr.eu-north-1.amazonaws.com

echo "Pulling Docker Image..."
docker pull 252312374343.dkr.ecr.eu-north-1.amazonaws.com/yt-plugin-ecr:latest


echo "Checking Existing Container..."
if [ "$(docker ps -q -f name=yt_plugin_container)" ]; then
    echo "Stopping Existing Container..."   
    docker stop yt_plugin_container || true
fi
if [ "$(docker ps -aq -f name=yt_plugin_container)" ]; then
    echo "Removing Existing Container..."
    docker rm yt_plugin_container || true
fi

echo "Starting New Container..."
docker run -d -p 80:5000 --name yt_plugin_container 252312374343.dkr.ecr.eu-north-1.amazonaws.com/yt-plugin-ecr:latest

echo "Container Started Successfully."




