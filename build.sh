#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
    cargo build --features "metal"
elif [[ "$OSTYPE" == "cygwin" ]]; then
    cargo build --features "vulkan"
elif [[ "$OSTYPE" == "msys" ]]; then
    cargo build --features "vulkan"
elif [[ "$OSTYPE" == "win32" ]]; then
    cargo build --features "vulkan"
fi