# gunicorn_config.py
bind = "0.0.0.0:5000"
workers = 1
threads = 2
timeout = 120