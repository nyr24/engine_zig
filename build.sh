#/bin/bash

DEBUG_BUILD_DIR="build/debug"
RELEASE_BUILD_DIR="build/release"
ASSETS_SRC_DIR="assets"

BUILD_MODE="Debug";
BUILD_TESTS=0;
COPY_ASSETS=0;
VSYNC=0;
SANITIZE=0;
ZIG_OPTS=""
ZIG_CUSTOM_OPTS=""
CMAKE_OPTS=""
CMAKE_BUILD_OPTS="-j"

if [ "$XDG_SESSION_TYPE" == "wayland" ]; then
    echo "Current session is Wayland."
    ZIG_CUSTOM_OPTS+=" -Dlinux_session=wl"
  elif [ "$XDG_SESSION_TYPE" == "x11" ]; then
    echo "Current session is X11."
    ZIG_CUSTOM_OPTS+=" -Dlinux_session=x11"
else
    echo "Current session is Windows."
fi

for arg in "$@"; do
  case "$arg" in
  -r | --release)
    BUILD_MODE="ReleaseFast"
    ;;
  -rf | --releaseFast)
    BUILD_MODE="ReleaseFast"
    ;;
  -rs | --releaseSafe)
    BUILD_MODE="ReleaseSafe"
    ;;
  -rm | --releaseSmall)
    BUILD_MODE="ReleaseSmall"
    ;;
  -san | --sanitize)
    echo "Building with sanitizer"
    SANITIZE=1;
    ;;
  -c | --cleanup)
    echo "Rebuilding"
    ZIG_OPTS+=" new"
    CMAKE_BUILD_OPTS+=" --clean-first"
  --vsync)
    echo "Vsync enabled"
    ZIG_CUSTOM_OPTS+=" -Dvsync=1"
    ;;
  --test | -t)
    echo "Building tests..."
    BUILD_TESTS=1
    ;;
  -as | --assets)
    echo "Copying assets"
    COPY_ASSETS=1
    ;;
  *)
    echo "Unknown argument: $arg"
    ;;
  esac
done

[ -d "build" ] || mkdir "build"
[ -d "assets" ] || mkdir "assets"

echo "Building in $BUILD_MODE mode"

if [ $BUILD_MODE == "Debug" ]; then
  CMAKE_OPTS+=" -DCMAKE_BUILD_TYPE=Debug"
  [ -d "$DEBUG_BUILD_DIR" ] || mkdir "$DEBUG_BUILD_DIR"
  # assets
  if [ $COPY_ASSETS -eq 1 ]; then
    [ -d "$DEBUG_BUILD_DIR/assets" ] || mkdir -p "$DEBUG_BUILD_DIR/assets"
    cp -r "$ASSETS_SRC_DIR" "$DEBUG_BUILD_DIR/assets"
  fi
  cd "$DEBUG_BUILD_DIR"
  cmake $CMAKE_OPTS ../../ && cmake --build . $CMAKE_BUILD_OPTS
  zig build $ZIG_OPTS -- -p $DEBUG_BUILD_DIR -Doptimize=$BUILD_MODE $ZIG_CUSTOM_OPTS
else
  CMAKE_OPTS+=" -DCMAKE_BUILD_TYPE=Release"
  [ -d "$RELEASE_BUILD_DIR" ] || mkdir "$RELEASE_BUILD_DIR"
  # assets
  if [ $COPY_ASSETS -eq 1 ]; then
    [ -d "$RELEASE_BUILD_DIR/assets" ] || mkdir -p "$RELEASE_BUILD_DIR/assets"
    cp -r "$ASSETS_SRC_DIR" "$RELEASE_BUILD_DIR/assets"
  fi
  cd "$RELEASE_BUILD_DIR"
  cmake $CMAKE_OPTS ../../ && cmake --build . $CMAKE_BUILD_OPTS
  zig build $ZIG_OPTS -- -p $RELEASE_BUILD_DIR -Doptimize=$BUILD_MODE $ZIG_CUSTOM_OPTS
fi
