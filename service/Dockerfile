FROM python:3

ENV APP_HOME="/usr/src/app" \
    WEB_PORT="80"

WORKDIR ${APP_HOME}

COPY requirements.txt ${APP_HOME}

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r ${APP_HOME}/requirements.txt

COPY . ${APP_HOME}
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]

