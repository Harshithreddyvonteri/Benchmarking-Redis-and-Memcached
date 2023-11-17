# Redis Vs Memcached
This repository contains code for various experiments used to benchmark Redis and Memcached. We used memtier-benchmark to evaluate their performance.

## Project Structure
1. **src:** This directory contains the source code to run the experiments.
2. **logs:** The logs of execution will be stored here. A new run-id will be created for each run and the logs will be stored in the corresponsing directory.
3. **outputs:** All the plots and summary files are saved in this folder under the corresponding run-id of the execution.
4. **bin:** This folder contains the memtier-benchmarch binary used for performing experiments. Note that we changed the source code of memtier-benchmark. The modified source code of memter-benchmark can be found [todo]().
5. **assets:** This folder contains the configuration files of different persistance models of redis. These are used to evaluate the overhead of various persistance models.

## Requirements:
matplotlib, numpy, os, pandas, tqdm, sys, redis, pymemcache

## Execution
1. Before executing the project, make sure redis and memcached services are started. It is assumed that redis and memcached don't need authentication to connect. If required, please disable authentication or change the commands in ``src/run_experiments.py``. It is assumed that Redis runs on port 6379 and Memcached runs on port 11211.
2. ``cd`` into the project and Run ``python3 src/run_experiments.py [experiment_name]``
3. Available options for ``experiment_name``:
    1. ``latency_benchmark``: Runs latency benchmarks for redis, memcached with varying number of set, get operations and plots the average and 99-percentile latency. 
    2. ``throughput_benchmark``: Runs throughput benchmarks for redis, memcached and plots the throughput at each second of the execution.
    3. ``scalability_benchmark``: Runs scalability benchmark by varying number of threads and plots the average latency, average throughput.
    4. ``memory_usage:`` Runs memory usage benchmark by varying number of keys stored and stores the memory used.
    5. ``persistance_models:`` For this benchmark, you need to use the configuration file corresponding to the persistance model of interest (available in ``assets/`` dir) and restart redis. This experiment then saves the average latency and throughput corresponding to that persistance model. You need to do multiple runs to compare different persistance models.
4. The output plots and the csv files will be stored in ```outputs/{run-id}/```. The detailed logs of the execution can be found at ```logs/{run-id}/```.
5. Note: Don't remove ``logs/dump`` directory. It stores intermediate files during executions.

## References:
1. [Memtier-Benchmark](https://github.com/RedisLabs 
memtier_benchmark)