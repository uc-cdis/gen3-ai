# Metrics

By default, we support Prometheus metrics. They can be exposed at a `/metrics` endpoint compatible with Prometheus scraping and visualize in Prometheus or
Graphana, etc.

You can [run Prometheus locally](https://github.com/prometheus/prometheus) if you want to test or visualize these.

## Set Up Locally

Run the service locally using `just run {{service}}`.

Create a [`prometheus.yml` config file](https://prometheus.io/docs/prometheus/latest/configuration/configuration), such
as: `~/Documents/prometheus/conf/prometheus.yml`.

Put this in:

```yaml
global:
  scrape_interval: 15s # By default, scrape targets every 15 seconds.

# A scrape configuration containing exactly one endpoint to scrape:
# Here it's Prometheus itself.
scrape_configs:
  # The job name is added as a label `job=<job_name>` to any timeseries scraped from this config.
  - job_name: 'gen3_inference'

    # Override the global default and scrape targets from this job every 5 seconds.
    scrape_interval: 10s

    static_configs:
      # NOTE: The `host.docker.internal` below is so docker on MacOS can properly find the locally running service
      - targets: [ 'host.docker.internal:4143' ]

  - job_name: 'gen3_ai_model_repo'
    static_configs:
      - targets: [ 'host.docker.internal:4141' ]
  - job_name: 'gen3_embeddings'
    static_configs:
      - targets: [ 'host.docker.internal:4142' ]
```

> Note: Tested the above config on MacOS, with Linux you can maybe adjust these commands to actually expose the local
> network to the running prometheus container.

Then run this:

```
docker run --name prometheus -v ~/Documents/prometheus/conf/prometheus.yml:/etc/prometheus/prometheus.yml -d -p 127.0.0.1:9090:9090 prom/prometheus
```

Then go to [http://127.0.0.1:9090](http://127.0.0.1:9090).

And some recommended PromQL queries:

```promql
sum by (status_code) (gen3_inference_api_requests_total)
```
