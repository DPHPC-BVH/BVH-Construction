#!/bin/bash

NODES="1"
TIME="0:20:00"
CONSTRAINT="v100|a100"
COMMAND="./build/Src/main"

if [ "$#" -eq 1 ]; then
    COMMAND=$1
fi

# Load modules 
module load CMake/3.13.1
module load cuda
echo "--> Modules loaded"


# Compile 
echo "--> Start compilation"
cd build/
cmake ..
make
cd ..

echo "--> Submitting command: $COMMAND"

# Run slrum job
sbatch << EOF
#!/bin/sh

#SBATCH --nodes=$NODES
#SBATCH --time=$TIME
#SBATCH --constraint=$CONSTRAINT

$COMMAND

EOF