# omni-perception-3se

Minimal C++ perception pipeline managed by CMake.

This repo builds a small multi-threaded perception pipeline around an in-process 2D simulator:

- simulates a robot moving in a 2D scene with four fixed camera mounts
- simulates pedestrians walking along waypoint loops
- renders four synthetic camera frames each tick
- runs one mock-backed `yolo26-nano` detector instance per camera
- converts detections into robot-centric 2D observations
- fuses asynchronous observations into persistent 2D tracks
- prints fused state as line-delimited JSON

The point is simple: keep the pipeline runnable and testable without real cameras, GPU inference, physics engines, or ROS.

## What It Does

The executable [`omni-perception-3se`](src/main.cpp) simulates a four-camera rig:

- `left_front`
- `right_front`
- `left_rear`
- `right_rear`

Each camera gets its own queue and worker thread. A mock-backed `Yolo26NanoDetector` finds dark rectangular blobs in synthetic frames. Those detections are converted into approximate body-frame position reports, then a `FusionTracker` associates nearby observations into tracks such as `track-1`.

`StatePublisher` snapshots the latest fused state and prints compact JSON to stdout.

Typical output looks like this:

```json
{"sequence_id":42,"timestamp_ms":1036050526,"person":[{"track_id":"track-1","x_m":4.65,"y_m":2.48,"vx_mps":0.24,"vy_mps":-0.03,"range_m":5.27,"radius_m":0.57,"angle":28.1,"angle_velocity":-0.8,"confidence":0.91,"sources":["left_front","right_front"],"last_update_ms":1036050511}]}
```

The simulator starts with four walking pedestrians, so you should expect one to four fused tracks depending on robot pose and camera visibility.

## Project Layout

- [`src/main.cpp`](src/main.cpp): wires the end-to-end pipeline together
- [`src/simulation/simulation_engine.cpp`](src/simulation/simulation_engine.cpp): advances the robot/pedestrian world and renders camera images
- [`src/perception/yolo26_nano_detector.cpp`](src/perception/yolo26_nano_detector.cpp): mock-backed per-camera detector adapter
- [`src/perception/mock_person_detector.cpp`](src/perception/mock_person_detector.cpp): finds dark blobs inside synthetic images
- [`src/perception/camera_worker.cpp`](src/perception/camera_worker.cpp): runs per-camera detection and emits observations
- [`src/perception/fusion_tracker.cpp`](src/perception/fusion_tracker.cpp): associates 2D observations into tracks
- [`src/perception/state_publisher.cpp`](src/perception/state_publisher.cpp): publishes snapshots from the tracker
- [`src/perception/json_output.cpp`](src/perception/json_output.cpp): serializes pipeline state into JSON
- [`src/core/blocking_queue.hpp`](src/core/blocking_queue.hpp): tiny bounded queue used between threads
- [`tests/app_config_test.cpp`](tests/app_config_test.cpp): tests small config helpers
- [`tests/pipeline_test.cpp`](tests/pipeline_test.cpp): tests detection, threading, fusion, publishing, and JSON output

## Dependencies

This project uses `vcpkg` manifest mode.

Primary dependencies:

- `OpenCV`
- `fmt`
- `GoogleTest`

If `vcpkg` is not installed yet:

```bash
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh
export VCPKG_ROOT=~/vcpkg
```

## Configure

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
```

Or use the included `just` recipes:

```bash
just build
```

## Build

```bash
cmake --build build
```

Or:

```bash
just build
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

Or:

```bash
just test
```

Current tests cover:

- camera id parsing and grid helpers
- angle normalization and camera-angle conversion
- monocular range estimation
- mock detection behavior
- queue overwrite behavior
- camera worker message emission
- fusion of asynchronous 2D detections
- state publishing
- JSON serialization
- simulation stepping and rendering

## Run

```bash
./build/omni-perception-3se
```

Or:

```bash
just run
```

The program opens a small control window for keyboard input, writes JSON lines to stdout, and does not require physical cameras.

## Docker

Docker is useful here for reproducible configure/build/test runs.

Build the image:

```bash
docker build -t omni-perception-3se-dev .
```

Run a shell in the container:

```bash
docker run --rm -it -v "$PWD:/workspace" omni-perception-3se-dev
```

Build and test inside the container:

```bash
cmake -S . -B build-docker -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build-docker
ctest --test-dir build-docker --output-on-failure
```

## Notes

This is still intentionally small and fake in the right places:

- input frames are synthetic
- detection is a deterministic mock, not a real ML model
- fusion uses simple 2D nearest-neighbor association
- output is plain JSON, not ROS messages or a network API
- rerun integration is intentionally deferred for now

That is fine. The repo is set up to make pipeline behavior easy to test before swapping in real cameras or real detectors.
