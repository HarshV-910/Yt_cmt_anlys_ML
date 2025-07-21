# # gunicorn_config.py
# bind = "0.0.0.0:5000"
# workers = 2
# threads = 2
# timeout = 120

# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
# --------------------------------------------

# gunicorn_config.py
bind = "0.0.0.0:5000"
workers = 2
threads = 2
timeout = 120 # Increased timeout for potentially long API calls

worker_class = "gthread"