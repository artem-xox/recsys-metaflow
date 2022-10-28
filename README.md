# recsys-metaflow



---
### Launch

1. setting up environment
```shell
# install deps
pipenv install
pipenv shell

# fill envs
cp .env.template .env
```

2. configure metaflow
```shell
# fill the config and import it
metaflow configure import metaflow_config.json
```

2. run scripts
```shell
python recsys-metaflow/scripts/recommendations.py run
```

3. results analysis
```shell
jupyter notebook
```
