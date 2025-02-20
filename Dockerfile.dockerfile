FROM python:3.8-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y supervisor && apt-get clean

EXPOSE 8000 8001
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]