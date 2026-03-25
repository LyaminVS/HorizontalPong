#!/bin/bash
source credentials

docker rm -f ${CONTAINER_NAME} 2>/dev/null
echo "Starting container: ${CONTAINER_NAME}..."

# Запуск контейнера только на процессоре
docker run \
  -d \
  --shm-size=10g \
  --user ${DOCKER_USER_ID}:${DOCKER_GROUP_ID} \
  --name ${CONTAINER_NAME} \
  --rm \
  -v "$(pwd)":/app \
  -p 8890:8890 \
  ${DOCKER_NAME}

echo "Jupyter Notebook is running at: http://localhost:8890"
echo "Attaching to container shell..."
docker exec -it ${CONTAINER_NAME} /bin/bash