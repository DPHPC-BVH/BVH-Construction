#!/bin/bash

REPETITIONS=30
WARMUPS=2
RUNS=$((REPETITIONS + WARMUPS))
CLUSTER='true'

OUT_DIR="out/"
BENCH_OUT_DIR=$OUT_DIR"benchmarks/"
PLOTS_OUT_DIR=$OUT_DIR"plots/"

mkdir -p $PLOTS_OUT_DIR

# Create distribution and qq plots
declare -a distribution_plots=(
    "conference"
    "fairyforest"
    "sibenik"
    "sanmiguel"
)

for p in ${distribution_plots[@]};
do  
    FILE=${p}
    python3 ./plotter/plot.py distribution ${BENCH_OUT_DIR}${FILE}.csv --skip-first-n-iterations=${WARMUPS} --out=${PLOTS_OUT_DIR}${FILE}_distribution_with_labels.pdf
    python3 ./plotter/plot.py distribution ${BENCH_OUT_DIR}${FILE}.csv --without-labels --skip-first-n-iterations=${WARMUPS} --out=${PLOTS_OUT_DIR}${FILE}_distribution.pdf
    python3 ./plotter/plot.py qqplot ${BENCH_OUT_DIR}${FILE}.csv --skip-first-n-iterations=${WARMUPS} --out=${PLOTS_OUT_DIR}${FILE}_qqplot.pdf
done


# Create stage plots
python3 ./plotter/plot.py stages ${BENCH_OUT_DIR}sibenik_stages.csv ${BENCH_OUT_DIR}fairyforest_stages.csv ${BENCH_OUT_DIR}conference_stages.csv ${BENCH_OUT_DIR}sanmiguel_stages.csv --skip-first-n-iterations=${WARMUPS} --strategy=median --out=${PLOTS_OUT_DIR}plot_stages_median.pdf
python3 ./plotter/plot.py stages ${BENCH_OUT_DIR}sibenik_stages.csv ${BENCH_OUT_DIR}fairyforest_stages.csv ${BENCH_OUT_DIR}conference_stages.csv ${BENCH_OUT_DIR}sanmiguel_stages.csv --skip-first-n-iterations=${WARMUPS} --strategy=mean --out=${PLOTS_OUT_DIR}plot_stages_mean.pdf
python3 ./plotter/plot.py stages ${BENCH_OUT_DIR}sibenik_stages.csv ${BENCH_OUT_DIR}fairyforest_stages.csv ${BENCH_OUT_DIR}conference_stages.csv ${BENCH_OUT_DIR}sanmiguel_stages.csv --skip-first-n-iterations=${WARMUPS} --strategy=median --absolute --out=${PLOTS_OUT_DIR}plot_stages_median_absolute.pdf
python3 ./plotter/plot.py stages ${BENCH_OUT_DIR}sibenik_stages.csv ${BENCH_OUT_DIR}fairyforest_stages.csv ${BENCH_OUT_DIR}conference_stages.csv --skip-first-n-iterations=${WARMUPS} --strategy=median --absolute --out=${PLOTS_OUT_DIR}plot_stages_median_absolute_without_sanmiguel.pdf

python3 ./plotter/plot.py stages ${BENCH_OUT_DIR}sibenik_stages.csv ${BENCH_OUT_DIR}fairyforest_stages.csv ${BENCH_OUT_DIR}conference_stages.csv ${BENCH_OUT_DIR}sanmiguel_stages.csv --skip-first-n-iterations=${WARMUPS} --strategy=mean --absolute --out=${PLOTS_OUT_DIR}plot_stages_mean_absolute.pdf
python3 ./plotter/plot.py stages ${BENCH_OUT_DIR}sibenik_stages.csv ${BENCH_OUT_DIR}fairyforest_stages.csv ${BENCH_OUT_DIR}conference_stages.csv --skip-first-n-iterations=${WARMUPS} --strategy=mean --absolute --out=${PLOTS_OUT_DIR}plot_stages_mean_absolute_without_sanmiguel.pdf

# Create plots with optix baseline
python3 ./plotter/plot.py series ${BENCH_OUT_DIR}sibenik.csv ${BENCH_OUT_DIR}fairyforest.csv ${BENCH_OUT_DIR}conference.csv ${BENCH_OUT_DIR}sanmiguel.csv --baseline=plotter/optix_construction_baseline.txt --skip-first-n-iterations=${WARMUPS} --out=${PLOTS_OUT_DIR}baseline_comparison.pdf
python3 ./plotter/plot.py series ${BENCH_OUT_DIR}sibenik.csv ${BENCH_OUT_DIR}fairyforest.csv ${BENCH_OUT_DIR}conference.csv --baseline=plotter/optix_construction_baseline.txt --skip-first-n-iterations=${WARMUPS} --out=${PLOTS_OUT_DIR}baseline_comparison_without_sanmiguel.pdf
python3 ./plotter/plot.py series ${BENCH_OUT_DIR}sibenik.csv ${BENCH_OUT_DIR}fairyforest.csv ${BENCH_OUT_DIR}conference.csv ${BENCH_OUT_DIR}sanmiguel.csv --baseline=plotter/optix_construction_baseline.txt --skip-first-n-iterations=${WARMUPS} --with-ci --out=${PLOTS_OUT_DIR}baseline_comparison_with_ci.pdf
python3 ./plotter/plot.py series ${BENCH_OUT_DIR}sibenik.csv ${BENCH_OUT_DIR}fairyforest.csv ${BENCH_OUT_DIR}conference.csv --baseline=plotter/optix_construction_baseline.txt --skip-first-n-iterations=${WARMUPS} --with-ci --out=${PLOTS_OUT_DIR}baseline_comparison_with_ci_without_sanmiguel.pdf


# Create shared memory comparison plot
python3 ./plotter/plot.py series ${BENCH_OUT_DIR}sibenik_bounding_boxes.csv ${BENCH_OUT_DIR}fairyforest_bounding_boxes.csv ${BENCH_OUT_DIR}conference_bounding_boxes.csv ${BENCH_OUT_DIR}sanmiguel_bounding_boxes.csv ${BENCH_OUT_DIR}sibenik_bounding_boxes_shared_memory.csv ${BENCH_OUT_DIR}fairyforest_bounding_boxes_shared_memory.csv ${BENCH_OUT_DIR}conference_bounding_boxes_shared_memory.csv ${BENCH_OUT_DIR}sanmiguel_bounding_boxes_shared_memory.csv --series-labels withoutSharedMemory withSharedMemory --skip-first-n-iterations=${WARMUPS} --out=${PLOTS_OUT_DIR}shared_memory_comparison.pdf
python3 ./plotter/plot.py series ${BENCH_OUT_DIR}sibenik_bounding_boxes.csv ${BENCH_OUT_DIR}fairyforest_bounding_boxes.csv ${BENCH_OUT_DIR}conference_bounding_boxes.csv ${BENCH_OUT_DIR}sibenik_bounding_boxes_shared_memory.csv ${BENCH_OUT_DIR}fairyforest_bounding_boxes_shared_memory.csv ${BENCH_OUT_DIR}conference_bounding_boxes_shared_memory.csv --series-labels withoutSharedMemory withSharedMemory --skip-first-n-iterations=${WARMUPS} --out=${PLOTS_OUT_DIR}shared_memory_comparison_without_sanmiguel.pdf

python3 ./plotter/plot.py series ${BENCH_OUT_DIR}sibenik_bounding_boxes.csv ${BENCH_OUT_DIR}fairyforest_bounding_boxes.csv ${BENCH_OUT_DIR}conference_bounding_boxes.csv ${BENCH_OUT_DIR}sanmiguel_bounding_boxes.csv ${BENCH_OUT_DIR}sibenik_bounding_boxes_shared_memory.csv ${BENCH_OUT_DIR}fairyforest_bounding_boxes_shared_memory.csv ${BENCH_OUT_DIR}conference_bounding_boxes_shared_memory.csv ${BENCH_OUT_DIR}sanmiguel_bounding_boxes_shared_memory.csv --series-labels withoutSharedMemory withSharedMemory --skip-first-n-iterations=${WARMUPS} --with-ci --out=${PLOTS_OUT_DIR}shared_memory_comparison_with_ci.pdf
python3 ./plotter/plot.py series ${BENCH_OUT_DIR}sibenik_bounding_boxes.csv ${BENCH_OUT_DIR}fairyforest_bounding_boxes.csv ${BENCH_OUT_DIR}conference_bounding_boxes.csv ${BENCH_OUT_DIR}sibenik_bounding_boxes_shared_memory.csv ${BENCH_OUT_DIR}fairyforest_bounding_boxes_shared_memory.csv ${BENCH_OUT_DIR}conference_bounding_boxes_shared_memory.csv --series-labels withoutSharedMemory withSharedMemory --skip-first-n-iterations=${WARMUPS} --with-ci --out=${PLOTS_OUT_DIR}shared_memory_comparison_with_ci_without_sanmiguel.pdf
