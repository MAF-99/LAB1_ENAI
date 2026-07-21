# ============================================================
# FINAL Dockerfile — the result of NEXT_STEPS_DOCKER.md
# Don't start here! Build it yourself step by step in the guide;
# this file is the "solution" to compare against at the end.
# ============================================================

# Step 1: small official base image, pinned version
FROM python:3.12-slim

# Step 6: don't run as root
RUN useradd --create-home agent
WORKDIR /app

# Step 2: dependencies FIRST — this layer is cached until
# requirements.txt changes, so code edits rebuild in seconds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your completed skeleton (rename to app.py inside the image)
COPY skeleton_enterprise_multiagent.py app.py

# Step 3: config via env (overridable at `docker run`), NO secrets here
ENV LAB_STAGE=5 \
    MOCK=0 \
    REPORTS_DIR=/reports

# Step 4: the reports volume mount point
RUN mkdir /reports && chown agent /reports
VOLUME /reports

# Step 5: the API port
EXPOSE 8000

# Step 6: healthcheck hits the endpoint you built in the lab
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

USER agent
CMD ["python", "app.py", "serve"]
