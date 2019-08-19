#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
    cargo run --features "metal"
elif [[ "$OSTYPE" == "cygwin" ]]; then
    cargo run --features "vulkan"
elif [[ "$OSTYPE" == "msys" ]]; then
    cargo run --features "vulkan"
elif [[ "$OSTYPE" == "win32" ]]; then
    cargo run --features "vulkan"
fi