FROM python:3.12-slim

RUN useradd -m -u 1000 kerneluser

RUN pip install --no-cache-dir jupyter_client ipykernel pandas numpy openpyxl matplotlib

RUN python -m ipykernel install --sys-prefix

RUN mkdir /sandbox && chown kerneluser:kerneluser /sandbox

USER kerneluser
WORKDIR /sandbox

# (shell, iopub, stdin, hb, control)
EXPOSE 5555 5556 5557 5558 5559

ENTRYPOINT ["python", "-m", "ipykernel_launcher", "-f"]