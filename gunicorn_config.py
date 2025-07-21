# gunicorn_config.py
bind = "0.0.0.0:5000"
workers = 2
threads = 2
timeout = 120

worker_class = "gthread" # Ensure this matches your gunicorn command