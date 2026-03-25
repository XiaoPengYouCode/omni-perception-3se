# omni-perception-3se

Minimal C++ perception pipeline demo managed by CMake.

This repo is no longer just an OpenCV window sample. The current app builds a small multi-threaded pipeline that:

- generates four synthetic camera frames
- runs a mock person detector per camera
- converts detections into robot-centric angles
- fuses asynchronous observations into persistent tracks
- prints fused state as line-delimited JSON

The point is simple: keep the pipeline runnable and testable without real cameras, GPU inference, or ROS.

## What It Does

The executable [`omni-perception-3se`](/Users/flamingo/Projects/cmake-demo/src/main.cpp) simulates a four-camera rig:

- `left_front`
- `right_front`
- `left_rear`
- `right_rear`

Each camera gets its own queue and worker thread. A `MockPersonDetector` finds dark rectangular blobs in synthetic frames. Those detections are converted into angles around the robot body, then a `FusionTracker` associates nearby observations into tracks such as `track-1`.

`StatePublisher` snapshots the latest fused state and prints compact JSON to stdout.

Typical output looks like this:

```json
{"sequence_id":4,"timestamp_ms":1036050526,"person":[{"track_id":"track-1","angle":30.0,"angle_velocity":0.0,"confidence":1.00,"sources":["left_front"],"last_update_ms":1036050511},{"track_id":"track-2","angle":135.0,"angle_velocity":0.0,"confidence":1.00,"sources":["left_rear"],"last_update_ms":1036050521},{"track_id":"track-3","angle":-30.0,"angle_velocity":0.0,"confidence":1.00,"sources":["right_front"],"last_update_ms":1036050516}]}
```

Right now the demo frames contain three visible synthetic people across four cameras, so you should expect three fused tracks.

## Project Layout

- [`src/main.cpp`](/Users/flamingo/Projects/cmake-demo/src/main.cpp): wires the end-to-end demo pipeline together
- [`src/perception/demo_frames.cpp`](/Users/flamingo/Projects/cmake-demo/src/perception/demo_frames.cpp): creates synthetic camera images
- [`src/perception/mock_person_detector.cpp`](/Users/flamingo/Projects/cmake-demo/src/perception/mock_person_detector.cpp): detects dark blobs as fake people
- [`src/perception/camera_worker.cpp`](/Users/flamingo/Projects/cmake-demo/src/perception/camera_worker.cpp): runs per-camera detection and emits observations
- [`src/perception/fusion_tracker.cpp`](/Users/flamingo/Projects/cmake-demo/src/perception/fusion_tracker.cpp): associates observations into tracks
- [`src/perception/state_publisher.cpp`](/Users/flamingo/Projects/cmake-demo/src/perception/state_publisher.cpp): publishes snapshots from the tracker
- [`src/perception/json_output.cpp`](/Users/flamingo/Projects/cmake-demo/src/perception/json_output.cpp): serializes pipeline state into JSON
- [`src/core/blocking_queue.hpp`](/Users/flamingo/Projects/cmake-demo/src/core/blocking_queue.hpp): tiny bounded queue used between threads
- [`tests/app_config_test.cpp`](/Users/flamingo/Projects/cmake-demo/tests/app_config_test.cpp): tests small config helpers
- [`tests/pipeline_test.cpp`](/Users/flamingo/Projects/cmake-demo/tests/pipeline_test.cpp): tests detection, threading, fusion, publishing, and JSON output

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
- mock detection behavior
- queue overwrite behavior
- camera worker message emission
- fusion of asynchronous detections
- state publishing
- JSON serialization

## Run

```bash
./build/omni-perception-3se
```

Or:

```bash
just run
```

The program writes JSON lines to stdout. It does not open a GUI window and does not require physical cameras.

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
- fusion uses simple angular nearest-neighbor association
- output is plain JSON, not ROS messages or a network API

That is fine. The repo is set up to make pipeline behavior easy to test before swapping in real cameras or real detectors.
