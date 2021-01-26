FROM python:3.7

WORKDIR /ml-vis

COPY data        /ml-vis/data
COPY names       /ml-vis/names
COPY pip         /ml-vis/pip
COPY predictions /ml-vis/predictions
COPY vis_scripts /ml-vis/vis_scripts
COPY utils.py    /ml-vis

RUN pip install -r pip/requirements_deploy.txt

ENV PYTHONPATH="/ml-vis:$PYTHONPATH"

ENTRYPOINT ["bokeh", "serve", "vis_scripts/interactive.py"]
