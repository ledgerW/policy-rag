FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Copy Chainlit app files into app/
COPY chainlit /app
COPY policy_rag /app/policy_rag

RUN pip install -r /app/requirements.txt		

# Create a non-root user
RUN useradd -m myuser

# Change ownership of the /app directory to the non-root user
RUN chown -R myuser:myuser /app

# For local testing only. Not recognized in Heroku environment
EXPOSE 8000

# Switch to the non-root user
USER myuser

# Run the app. $PORT is set by Heroku
CMD python -m chainlit run app.py -h --host 0.0.0.0 --port $PORT