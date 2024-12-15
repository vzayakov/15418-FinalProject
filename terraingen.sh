#!/bin/bash

# Usage/help function
function usage() {
    echo "USAGE: ${BASH_SOURCE[0]} -h <height> -w <width> -f <filepath> -s <scale> -l <lacunarity> -p <persistence> -o <octaves>"
    echo
}

# Parse command-line arguments
while getopts :h::w::f::s::l::p::o: flag
do
    case "${flag}" in
        h) height=${OPTARG};;
        w) width=${OPTARG};;
        f) filepath=${OPTARG};;
        s) scale=${OPTARG};;
        l) lacunarity=${OPTARG};;
        p) persistence=${OPTARG};;
        o) octaves=${OPTARG};;
        \?) echo "Invalid option: -$OPTARG"; exit 1;;
        :) echo "Option -$OPTARG requires an argument"; exit 1;;
    esac
done

# Check that all mandatory arguments are set
for arg in height width filepath scale lacunarity persistence octaves; do
    if [[ -z "${!arg}" ]]; then
        echo "Missing required argument for ${arg}"
        usage
        exit 1
    fi
done

# Try to make the executable
make
# Check for errors
if [ $? -ne 0 ]; then
    echo "Failed to make executable"
    exit 1
fi

# If successful, run executable with command-line args
./terraingen -h $height -w $width -f $filepath -s $scale -l $lacunarity -p $persistence -o $octaves
