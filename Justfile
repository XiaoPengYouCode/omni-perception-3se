set shell := ["sh", "-cu"]

# List available commands by default.
default:
  @just --list

# Build a target. Defaults to `omni-perception-3se`. Pass a target name and/or `-r` or `--release`.
build *args:
  mode=Debug; dir=build; target=omni-perception-3se; \
  for arg in {{args}}; do \
    if [ "$arg" = "--release" ] || [ "$arg" = "-r" ]; then \
      mode=Release; dir=build-release; \
    elif [ "$target" = "omni-perception-3se" ]; then \
      target="$arg"; \
    fi; \
  done; \
  cmake -S . -B "$dir" \
    -DCMAKE_BUILD_TYPE="$mode" \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_TOOLCHAIN_FILE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake" \
    -DVCPKG_MANIFEST_MODE=ON; \
  cmake --build "$dir" --target "$target"

# Build and run a target. Defaults to `omni-perception-3se`. Pass a target name and/or `-r` or `--release`.
run *args:
  mode=Debug; dir=build; target=omni-perception-3se; \
  for arg in {{args}}; do \
    if [ "$arg" = "--release" ] || [ "$arg" = "-r" ]; then \
      mode=Release; dir=build-release; \
    elif [ "$target" = "omni-perception-3se" ]; then \
      target="$arg"; \
    fi; \
  done; \
  cmake -S . -B "$dir" \
    -DCMAKE_BUILD_TYPE="$mode" \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_TOOLCHAIN_FILE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake" \
    -DVCPKG_MANIFEST_MODE=ON; \
  cmake --build "$dir" --target "$target"; \
  "./$dir/$target"

# Build the Debug target and run the test suite.
test:
  cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_TOOLCHAIN_FILE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake" \
    -DVCPKG_MANIFEST_MODE=ON; \
  cmake --build build; \
  ctest --test-dir build --output-on-failure
