# Benchmarking with Airspeed Velocity (ASV)

Benchmark is performed using the module `asv`: 
- [Documentation](https://asv.readthedocs.io/en/stable/index.html),
- [GitHub](https://github.com/airspeed-velocity/asv).


### Running benchmarks

`asv` is working using the configuration file `asv.conf.json` and it is predefined to evaluate available benchmarks between two commits of the `HEAD` branch and the `master` branch.

To run it, execute: 
- `asv run`: run benchmarks on the two last commits of the `HEAD` branch and the `master` branch,
- `asv run [COMMIT_HASH1]^!`: run on the commit specified with its commit hash.

The first time that `asv run` is executed, machine information is collected into `~/.asv-machine.json` to make abstraction to machine performance. 
If always press `Enter`, information is automatically retrieved. 

Once the benchmark evaluated, results are collected in `./.asv/results/[MACHINE_NAME]/.`

### Comparing benchmarks 

Benchmarks already performed between two commits can be compared using `asv compare [COMMIT_HASH1] [COMMIT_HASH2]`.
A table is displayed showing the time of each benchmark for the two commits as well as the ratio of the two times.
Be careful to indicate the oldest commit first to get appropriate ratio evaluating if any change has been beneficial or not. 

### Profiling 

For one benchmark and one commit, profiling may be performed using `asv profile [PATH.TO.BENCHMARK] [COMMIT_HASH]`.

### Writing benchmarks

All benchmarks are grouped in the folder `./benchmark_suite` in scripts. 
To summarize the writing of a benchmark: 
- can be written as function or as a class method,
- is automatically detected by the `time` prefix pattern in the function or the class method,
- a class can have a setup initialisation to prepare benchmark function and avoid out-of-scope operations to be timed,
- in a class, the attributes `number` and `repeat` may be set to make time statistics.





