# cmake-demo

Minimal C++ project managed by CMake.

This project starts with a small but sane foundation:

- `OpenCV` for camera capture and rendering
- `fmt` for structured console logging
- `GoogleTest` for early unit tests
- `vcpkg` manifest mode for dependency management

The current app version creates a simple image with OpenCV, shows it in a window, and logs progress with fmt.

## Docker

Use Docker to lock down the build and test environment.

For this project, Docker is good at:

- reproducing dependencies
- building the project the same way every time
- running unit tests in a clean environment

Docker is not the best place for your main camera-app loop on macOS, because camera access and GUI display get annoying fast.
So the clean split is:

- native macOS for camera capture and live display
- Docker for build and test

Build the image:

```bash
docker build -t cmake-demo-dev .
```

Run a shell in the container:

```bash
docker run --rm -it -v "$PWD:/workspace" cmake-demo-dev
```

Build and test inside the container:

```bash
cmake -S . -B build-docker -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build-docker
ctest --test-dir build-docker --output-on-failure
```

Default behavior:

- Create a simple image in memory
- Show it in an OpenCV window
- Print logs through `fmt` helpers
- Press `q` or `Esc` to quit

## Dependency

This project uses `vcpkg` manifest mode.

If you do not have `vcpkg` yet:

```bash
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh
export VCPKG_ROOT=~/vcpkg
```

## Configure

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
```

## Build

```bash
cmake --build build
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

## Run

```bash
./build/cmake_demo
```


## Files

- `vcpkg.json`: declares the OpenCV dependency
- `CMakePresets.json`: preset for `vcpkg`
- `src/logger.hpp`: small `fmt`-based logging helpers
- `tests/app_config_test.cpp`: unit tests for small parsing/grid helpers

## Notes

CI/CD can wait a bit.

Right now the useful move is:

- keep the build reproducible with `vcpkg`
- keep logic testable outside camera hardware
- add more tests as parsing, config, and frame-processing logic grows
