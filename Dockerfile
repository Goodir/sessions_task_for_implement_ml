FROM tiangolo/uwsgi-nginx-flask:python3.11
COPY ./app /app
COPY ./requirements.txt ./
COPY ./uwsgi.ini /app/uwsgi.ini
COPY ./model /model
RUN pip install --no-cache-dir -r requirements.txt