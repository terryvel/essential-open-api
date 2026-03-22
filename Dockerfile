FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y default-jre && rm -rf /var/lib/apt/lists/*

# Enable and install SSH for Azure
# ssh root@127.0.0.1 -p 2222 -c aes256-cbc
ENV SSH_PASSWD "root:Docker!"
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-server \
    && echo "$SSH_PASSWD" | chpasswd 

COPY sshd_config /etc/ssh/

ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV FLASK_APP=wsgi.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5100

EXPOSE 5100

# CMD ["flask", "run"]
COPY startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh

ENTRYPOINT ["/usr/local/bin/startup.sh"]
