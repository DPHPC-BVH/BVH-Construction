# Minimum Requirements

- C++17
- CMake 3.13
- CUDA 11.0


# Build

```bash
# Unzip the Scenes 
unzip scenes.zip

# Generate MakeFiles with CMake
mkdir build/
cd build/
cmake ..

# Compile
make
```

# Run

## Application

```bash
./build/Src/main --help
```

## UnitTests

To run all tests
```bash
./build/Test/test_dphp_bvh
```

For advanced use look at
```bash
./build/Test/test_dphp_bvh --help
```

## Benchmarks

To run all benchmarks

```
./build/Bench/bench_dphp_bvh --benchmark_repetitions=10 --benchmark_counters_tabular=true
```
where you can change ```--benchmark_repetitions``` to increase or decrease the number of repetitions


To run a single benchmark use
```
./build/Bench/bench_dphp_bvh --benchmark_filter=<name of benchmark> --benchmark_repetitions=10 --benchmark_counters_tabular=true
```

For advanced use look at
```bash
./build/Bench/bench_dphp_bvh --help
```

## Collect benchmark data and make plots

First install the python dependencies. For instance 
```
pip install -r requirements.txt
```

Then compile the project and execute the scripts `collect_benchmark_data.sh` and `make_all_plots.sh`
