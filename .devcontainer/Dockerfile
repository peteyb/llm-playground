FROM python:3.12-bookworm

ARG DEV_USER=playground

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=0 \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYDEVD_DISABLE_FILE_VALIDATION=1 \
    PATH="/home/$DEV_USER/.local/bin:${PATH}" \
    PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring

RUN adduser --disabled-password --uid 1000 --gecos '' $DEV_USER \
    && adduser $DEV_USER sudo && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && chown -R $DEV_USER:$DEV_USER /usr \
    && apt-get update \
    && apt-get install -y libpq-dev \
    && apt-get install -y bash-completion git vim

USER $DEV_USER:$DEV_USER

WORKDIR /workspace/app
COPY --chown=$DEV_USER:$DEV_USER app/pyproject.toml app/poetry.lock ./
RUN pip install -U pip \
    && pip install 'poetry==1.8.4' \
    && poetry install --no-interaction --no-ansi \
    && rm pyproject.toml poetry.lock

COPY .bash_history /home/$DEV_USER
VOLUME /workspace