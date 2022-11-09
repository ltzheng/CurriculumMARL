# CurriculumMARL

## Start with Docker

```shell
docker build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t spc -f Dockerfile .
./run_experiments.sh
```

## Cite Our Paper
