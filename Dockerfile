# Используем более свежий образ, подходящий для PyTorch 2.0+
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Отключаем интерактивные запросы при установке пакетов
ARG DEBIAN_FRONTEND=noninteractive

# Устанавливаем базовые утилиты и зависимости для Pygame/OpenCV
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ffmpeg \
    libsm6 \
    libxext6 \
    unzip \
    tmux \
    libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

# Рабочая директория
RUN mkdir /app
WORKDIR /app

# Аргументы для создания пользователя
ARG DOCKER_NAME
ARG DOCKER_USER_ID
ARG DOCKER_GROUP_ID

USER root
# Создаем группу и пользователя с правами хоста (чтобы не было проблем с правами на файлы)
RUN groupadd -g $DOCKER_GROUP_ID $DOCKER_NAME || true
RUN adduser --disabled-password --uid $DOCKER_USER_ID --gid $DOCKER_GROUP_ID --gecos '' --shell /bin/bash $DOCKER_NAME \
 && chown -R $DOCKER_NAME:$DOCKER_NAME /app

# Даем sudo без пароля
RUN echo "$DOCKER_NAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-$DOCKER_NAME
USER $DOCKER_NAME

ENV HOME=/home/$DOCKER_NAME
RUN mkdir $HOME/.cache $HOME/.config \
 && chmod -R 777 $HOME

# Копируем requirements (в build.sh мы делаем временную копию)
COPY --chown=$DOCKER_NAME:$DOCKER_NAME requirements.txt /app/requirements.txt

# Настройка Conda (берем latest для Python 3.10+)
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=$HOME/miniconda/bin:$PATH

RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && pip install --no-cache-dir -r /app/requirements.txt \
 && conda clean -ya

# Отключаем вывод графики Pygame по умолчанию (чтобы не крашилось при обучении без монитора)
ENV SDL_VIDEODRIVER=dummy

# Запускаем Jupyter Notebook по умолчанию
CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--port=8890", "--NotebookApp.token=''", "--NotebookApp.password=''"]