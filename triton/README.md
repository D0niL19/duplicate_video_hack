```
docker run -it --shm-size=1G --rm  \
    -p8000:8000 -p8001:8001 -p8002:8002 \
    -v ${PWD}:/workspace/ -v ${PWD}/model_repository_main:/models \
    triton_server:latest
```