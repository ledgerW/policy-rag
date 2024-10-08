FROM python:3.11
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
#COPY --chown=user . $HOME/app
COPY --chown=user ./hf_space $HOME/app
COPY --chown=user ./policy_rag $HOME/app/policy_rag
#COPY ./requirements.txt ~/app/requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["chainlit", "run", "app.py", "--port", "7860"]