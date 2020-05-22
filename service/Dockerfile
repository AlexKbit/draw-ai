FROM python:3.7.0-slim

RUN mkdir /app
WORKDIR /app
ADD . .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

CMD python app.py --web_app_port $PORT

