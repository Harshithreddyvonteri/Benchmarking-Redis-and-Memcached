from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys
import os, shutil
from tqdm import tqdm
from redis.client import Redis
from pymemcache.client import base

def get_avg_latency():
    df = pd.read_csv(f'logs/dump/avg_latency_1.txt', header = None)
    return np.mean(df)

def get_99_latency():
    df = pd.read_csv(f'logs/dump/latency_percentile.txt', header = None)
    return np.mean(df)

def get_avg_throughput():
    df = pd.read_csv(f'logs/dump/avg_throughput_1.txt', header = None)
    return np.mean(df)

rds_client = Redis(host='localhost', port=6379, password='', db=0, decode_responses=False)
memcached_client = base.Client(('localhost', 11211))

experiment = sys.argv[1]
runs = os.listdir('logs/')
runs = [int(run) for run in runs if run.isnumeric()]
run_id = max(runs) + 1 if len(runs) > 0 else 0
os.mkdir(f'logs/{run_id}')
os.mkdir(f'outputs/{run_id}')

print(f"Run id: {run_id}")
print(f"Logs stored at: logs/{run_id}/")
print(f"Outputs stored at: outputs/{run_id}/")

if(experiment == 'latency_benchmark'):
    print("Running Latency Benchmark....")
    num_operations = [1000, 10000, 100000, 1000000]
    redis_get_avg = []
    redis_set_avg = []
    memcached_get_avg = []
    memcached_set_avg = []
    redis_get_99 = []
    redis_set_99 = []
    memcached_get_99 = []
    memcached_set_99 = []
    for i in tqdm(range(len(num_operations))):
        rds_client.flushall()
        memcached_client.flush_all()
        opr = num_operations[i]

        # Redis Set
        redis_set_command = f"./bin/memtier_benchmark --hide-histogram -t 1 -c 1 --ratio 1:0 -n {opr} >> logs/{run_id}/cout.txt 2>&1"
        os.system(redis_set_command)
        redis_set_avg.append(get_avg_latency())
        redis_set_99.append(get_99_latency())

        # Redis Get
        redis_get_command = f"./bin/memtier_benchmark --hide-histogram -t 1 -c 1 --ratio 0:1 -n {opr} >> logs/{run_id}/cout.txt 2>&1"
        os.system(redis_get_command)
        redis_get_avg.append(get_avg_latency())
        redis_get_99.append(get_99_latency())

        # Memcached Set
        memcached_set_command = f"./bin/memtier_benchmark --hide-histogram -t 1 -c 1 --ratio 1:0 -P memcache_binary -p 11211 -n {opr} >> logs/{run_id}/cout.txt 2>&1"
        os.system(memcached_set_command)
        memcached_set_avg.append(get_avg_latency())
        memcached_set_99.append(get_99_latency())

        # Memcached Get
        memcached_get_command = f"./bin/memtier_benchmark --hide-histogram -t 1 -c 1 --ratio 0:1 -P memcache_binary -p 11211 -n {opr} >> logs/{run_id}/cout.txt 2>&1"
        os.system(memcached_get_command)
        memcached_get_avg.append(get_avg_latency())
        memcached_get_99.append(get_99_latency())

    # Plot Avg Latencies
    plt.plot(num_operations, redis_set_avg, label = 'Redis-Set')
    plt.plot(num_operations, memcached_set_avg, label = 'Memcached-Set')
    plt.plot(num_operations, redis_get_avg, label = 'Redis-Get')
    plt.plot(num_operations, memcached_get_avg, label = 'Memcached-Get')
    plt.xlabel('Num Operations')
    plt.ylabel('Avg Latency (msec)')
    plt.title('Latency Benchmark')
    plt.semilogx()
    plt.legend()
    plt.xticks([1000, 10000, 100000, 1000000])
    plt.savefig(f'outputs/{run_id}/avg_latency_benchmark.png')

    # Plot 99 percentile Latencies
    plt.clf()
    plt.plot(num_operations, redis_set_99, label = 'Redis-Set')
    plt.plot(num_operations, memcached_set_99, label = 'Memcached-Set')
    plt.plot(num_operations, redis_get_99, label = 'Redis-Get')
    plt.plot(num_operations, memcached_get_99, label = 'Memcached-Get')
    plt.xlabel('Num Operations')
    plt.ylabel('p99 Latency (msec)')
    plt.title('Latency Benchmark')
    plt.semilogx()
    plt.legend()
    plt.xticks([1000, 10000, 100000, 1000000])
    plt.savefig(f'outputs/{run_id}/p99_latency_benchmark.png')

    # Output the latency values
    df = pd.DataFrame()
    df['num_operations'] = num_operations
    df['redis-set-avg'] = redis_set_avg
    df['memcached-set-avg'] = memcached_set_avg
    df['redis-get-avg'] = redis_get_avg
    df['memcached-get-avg'] = memcached_get_avg
    df['redis-set-p99'] = redis_set_99
    df['memcached-set-p99'] = memcached_set_99
    df['redis-get-p99'] = redis_get_99
    df['memcached-get-p99'] = memcached_get_99
    df.to_csv(f'outputs/{run_id}/latency_values.csv', index = False)

elif(experiment == 'throughput_benchmark'):
    print("Running Throughput Benchmark....")

    rds_client.flushall()
    memcached_client.flush_all()
    
    # Redis command
    redis_command = f"./bin/memtier_benchmark --hide-histogram -t 4 -c 50 -n 1000000 -d 10  >> logs/{run_id}/cout.txt 2>&1"
    os.system(redis_command)
    df_redis = pd.read_csv(f'logs/dump/throughput_1.txt', header = None)

    # Memcached command
    memcached_command = f"./bin/memtier_benchmark --hide-histogram -t 4 -c 50 -P memcache_binary -p 11211 -n 1000000 -d 10  >> logs/{run_id}/cout.txt 2>&1"
    os.system(memcached_command)
    df_memcache = pd.read_csv(f'logs/dump/throughput_1.txt', header = None)

    # Plot Throughput
    plt.plot(df_redis/1e5, label = 'Redis')
    plt.plot(df_memcache/1e5, label = 'Memcached')
    plt.xlabel('Time (sec)')
    plt.ylabel('Throughput (ops/sec) in 1e5')
    plt.title('Throughput Benchmark')
    plt.legend()
    plt.savefig(f'outputs/{run_id}/throughput_benchmark.png')

    # Save Results
    df = pd.DataFrame()
    df['time (in sec)'] = [i for i in range(max(len(df_memcache), len(df_redis)))]
    df['redis-throughput'] = df_redis
    df['memcached-throughput'] = df_memcache
    df.to_csv(f'outputs/{run_id}/throughputs.csv', index = False)

elif(experiment == 'scalability_benchmark'):
    print("Running Scalability Benchmark....")
    num_threads = [1, 2, 4, 8, 16]
    redis_latency_avg = []
    memcached_latency_avg = []
    redis_throughput_avg = []
    memcached_throughput_avg = []
    for i in tqdm(range(len(num_threads))):
        rds_client.flushall()
        memcached_client.flush_all()
        t = num_threads[i]

        # Redis Command
        redis_command = f"./bin/memtier_benchmark --hide-histogram -t {t} -c 50 -n 10000 >> logs/{run_id}/cout.txt 2>&1"
        os.system(redis_command)
        redis_latency_avg.append(get_avg_latency())
        redis_throughput_avg.append(get_avg_throughput())

        # Memcached Command
        memcached_command = f"./bin/memtier_benchmark --hide-histogram -t {t} -c 50 -P memcache_binary -p 11211 -n 10000 >> logs/{run_id}/cout.txt 2>&1"
        os.system(memcached_command)
        memcached_latency_avg.append(get_avg_latency())
        memcached_throughput_avg.append(get_avg_throughput())

    # Plot Latency
    plt.plot(num_threads, np.array(redis_latency_avg), label = 'Redis')
    plt.plot(num_threads, np.array(memcached_latency_avg), label = 'Memcached')
    plt.xlabel('Num threads')
    plt.ylabel('Avg Latency (msec)')
    plt.title('Scalability Benchmark')
    plt.legend()
    plt.xticks([1, 2, 4, 8, 16])
    plt.savefig(f'outputs/{run_id}/scalability_benchmark_latency.png')

    # Plot Throughput
    plt.clf()
    plt.plot(num_threads, np.array(redis_throughput_avg)/1e5, label = 'Redis')
    plt.plot(num_threads, np.array(memcached_throughput_avg)/1e5, label = 'Memcached')
    plt.xlabel('Num threads')
    plt.ylabel('Avg Throughput (ops/sec) in 1e5')
    plt.title('Scalability Benchmark')
    plt.legend()
    plt.xticks([1, 2, 4, 8, 16])
    plt.savefig(f'outputs/{run_id}/scalability_benchmark_throughput.png')

    # Save the values
    df = pd.DataFrame()
    df['num_threads'] = num_threads
    df['redis-latency'] = redis_latency_avg
    df['memcached-latency'] = memcached_latency_avg
    df['redis-throughput'] = redis_throughput_avg
    df['memcached-throughput'] = memcached_throughput_avg
    df.to_csv(f'outputs/{run_id}/scalability.csv', index = False)

elif(experiment == 'persistance_models'):
    print("Running Persistance Models Benchmark....")
    redis_command = f"./bin/memtier_benchmark --hide-histogram -t 4 -c 50 -n 100000  >> logs/{run_id}/cout.txt 2>&1"
    os.system(redis_command)
    avg_latency = get_avg_latency()
    avg_throughput = get_avg_throughput()

    # Save results
    df = pd.DataFrame()
    df['avg_latency'] = avg_latency
    df['avg_throughput'] = avg_throughput
    df.to_csv(f'outputs/{run_id}/avg_latency_throughput.csv', index = False)

elif(experiment == 'memory_usage'):
    print("Running Memory Usage benchmark....")
    num_keys = [10000, 100000, 1000000]
    redis_memory = []
    memcached_memory = []
    for i in tqdm(range(len(num_keys))):
        rds_client.flushall()
        memcached_client.flush_all()
        k = num_keys[i]
        for s in range(k):
            rds_client.set(str(s), str(s))
            memcached_client.set(str(s), str(s))
        redis_stats = rds_client.memory_stats()
        memcached_stats = memcached_client.stats()
        redis_memory.append(redis_stats['total.allocated'] - redis_stats['startup.allocated'])
        memcached_memory.append(memcached_stats[b'bytes'])
    
    df = pd.DataFrame()
    df['num_keys'] = num_keys
    df['redis-memory (in bytes)'] = redis_memory
    df['memcached-memory (in bytes)'] = memcached_memory
    df.to_csv(f'outputs/{run_id}/memory_usage.csv', index = False)

elif (experiment == 'clean'):
    shutil.rmtree(f'logs/{run_id}')
    shutil.rmtree(f'logs/{run_id-1}')
    shutil.rmtree(f'outputs/{run_id}')
    shutil.rmtree(f'outputs/{run_id-1}')
    shutil.rmtree(f'logs/dump')
    os.mkdir('logs/dump/')
else:
    print("Invalid Experiment")
# memtier_benchmark --hide-histogram -t 1 -c 1 --ratio 1:0 -P memcache_binary -p 11211 -n 10000