# The match runs this with no network, a read-only disk (writable /tmp), 1 CPU, 512 MB, and the map at /arena.
FROM python:3.13-slim
WORKDIR /bot
COPY mechfight.py nav.py bot.py nav.bin ./
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
