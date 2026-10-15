#!/usr/bin/env bash
# Build the pinned MiniCPM-V runtime used by the optional benchmark package.
# Requires Xcode and CMake >= 3.28 (CMAKE=/path/to/cmake is supported).
# Usage: scripts/prepare_minicpm.sh [all|macos|ios|ios-device|ios-sim]
# Generated sources, logs and binaries stay in ignored directories.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${ROOT}/build/minicpm"
DEST="${ROOT}/MiniCPMBenchmark/Frameworks/llama.xcframework"
# Runtime pinned by OpenBMB/MiniCPM-V-Apps at:
# 68f53d67a418dd6444659c39005dd5cc4097071c
REVISION=96bbf61710d44421ea54e63f5194f9b17c20d877
ARCHIVE_SHA256=5c8c3a490cb7ae216c24a1bdcdddcd1d048b28d90285978d3ca7205c9a7ff8b0
SOURCE="${WORK}/llama.cpp-omni-${REVISION}"
ARCHIVE="${WORK}/runtime.tar.gz"
LOCAL_PATCH_ID=minicpm-metal-worker-autorelease-v1
LOCAL_PATCH="${ROOT}/scripts/patches/minicpm-metal-worker-autorelease.patch"
MODE="${1:-all}"
IOS_MIN=17.0
MACOS_MIN=14.0
CMAKE="${CMAKE:-cmake}"
JOBS="${JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || printf 4)}"

case "${MODE}" in all|macos|ios|ios-device|ios-sim) ;; *)
    echo "Usage: $0 [all|macos|ios|ios-device|ios-sim]" >&2; exit 2 ;;
esac
command -v xcrun >/dev/null || { echo 'Xcode is required.' >&2; exit 1; }
command -v "${CMAKE}" >/dev/null || {
    echo 'CMake >= 3.28 is required; install it or set CMAKE=/absolute/path/to/cmake.' >&2
    exit 1
}
CMAKE_VERSION="$("${CMAKE}" --version | head -n 1 | awk '{print $3}')"
CMAKE_MAJOR="${CMAKE_VERSION%%.*}"
CMAKE_REST="${CMAKE_VERSION#*.}"
CMAKE_MINOR="${CMAKE_REST%%.*}"
if (( CMAKE_MAJOR < 3 || (CMAKE_MAJOR == 3 && CMAKE_MINOR < 28) )); then
    echo "CMake ${CMAKE_VERSION} is too old; version 3.28 or later is required." >&2
    exit 1
fi
mkdir -p "${WORK}" "$(dirname "${DEST}")"

if [[ ! -f "${ARCHIVE}" ]]; then
    curl --fail --location --retry 3 \
        "https://codeload.github.com/tc-mb/llama.cpp-omni/tar.gz/${REVISION}" \
        -o "${ARCHIVE}.partial"
    mv "${ARCHIVE}.partial" "${ARCHIVE}"
fi
ACTUAL_SHA256="$(shasum -a 256 "${ARCHIVE}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${ARCHIVE_SHA256}" ]]; then
    echo "Runtime archive checksum mismatch: ${ARCHIVE}" >&2
    exit 1
fi
if [[ ! -f "${SOURCE}/CMakeLists.txt" ]]; then
    tar -xzf "${ARCHIVE}" -C "${WORK}"
fi

# Pinned upstream plus one explicit ownership-only patch. Forward/reverse dry
# runs make repeated builds idempotent and reject an unexpected source layout.
LOCAL_PATCH_SHA256="$(shasum -a 256 "${LOCAL_PATCH}" | awk '{print $1}')"
if patch -C -f -F 0 -p1 -d "${SOURCE}" -i "${LOCAL_PATCH}" >/dev/null 2>&1; then
    patch -f -F 0 -p1 -d "${SOURCE}" -i "${LOCAL_PATCH}"
elif patch -C -f -F 0 -R -p1 -d "${SOURCE}" -i "${LOCAL_PATCH}" >/dev/null 2>&1; then
    echo "Local runtime patch already applied: ${LOCAL_PATCH_ID}"
else
    echo "Runtime source does not match either side of ${LOCAL_PATCH_ID}; refusing to build." >&2
    exit 1
fi

# llama-common supplies upstream DSpark speculative decoding and sampling.
# Enabling TOOLS exposes mtmd; no omni, speech, server or CLI target is compiled.
COMMON_ARGS=(
    -G Xcode
    -DBUILD_SHARED_LIBS=OFF
    -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_UI=OFF
    -DLLAMA_OPENSSL=OFF
    -DLLAMA_BUILD_NUMBER=0 -DLLAMA_BUILD_COMMIT="${REVISION}"
    -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BLAS=ON
    -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY=
    -DCMAKE_OSX_ARCHITECTURES=arm64
)
XC_ARGS=()

build_slice() {
    local name="$1" sdk="$2" platform="$3" minimum="$4" system="$5"
    local dir="${WORK}/${name}" release=Release
    local fw="${WORK}/${name}/framework/llama.framework"
    local extra_args=() min_flag
    if [[ "${system}" == iOS ]]; then
        release="Release-${sdk}"
        extra_args=(-DCMAKE_SYSTEM_NAME=iOS -DIOS=ON
            -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS="${sdk}")
        if [[ "${sdk}" == iphoneos ]]; then
            min_flag="-mios-version-min=${minimum}"
        else
            min_flag="-mios-simulator-version-min=${minimum}"
        fi
    else
        min_flag="-mmacosx-version-min=${minimum}"
    fi
    mkdir -p "${dir}"
    echo "Configuring ${name}; logs: ${dir}/configure.log"
    if ! "${CMAKE}" -S "${SOURCE}" -B "${dir}" "${COMMON_ARGS[@]}" \
        "${extra_args[@]}" -DCMAKE_OSX_SYSROOT="${sdk}" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="${minimum}" >"${dir}/configure.log" 2>&1; then
        tail -n 60 "${dir}/configure.log" >&2; exit 1
    fi
    echo "Building ${name}; logs: ${dir}/build.log"
    if ! "${CMAKE}" --build "${dir}" --config Release --target llama mtmd llama-common \
        -j "${JOBS}" -- -quiet >"${dir}/build.log" 2>&1; then
        tail -n 80 "${dir}/build.log" >&2; exit 1
    fi

    # Statically merge the explicit dependency list, then expose it as a
    # framework dylib so SwiftPM/Xcode can embed and sign it on iOS.
    local libs=(
        "${dir}/src/${release}/libllama.a"
        "${dir}/tools/mtmd/${release}/libmtmd.a"
        "${dir}/common/${release}/libllama-common.a"
        "${dir}/common/${release}/libllama-common-base.a"
        "${dir}/vendor/cpp-httplib/${release}/libcpp-httplib.a"
        "${dir}/ggml/src/${release}/libggml.a"
        "${dir}/ggml/src/${release}/libggml-base.a"
        "${dir}/ggml/src/${release}/libggml-cpu.a"
        "${dir}/ggml/src/ggml-metal/${release}/libggml-metal.a"
        "${dir}/ggml/src/ggml-blas/${release}/libggml-blas.a"
    )
    mkdir -p "${fw}/Headers" "${fw}/Modules"
    xcrun libtool -static -o "${dir}/combined.a" "${libs[@]}" \
        >"${dir}/link.log" 2>&1
    xcrun --sdk "${sdk}" clang++ -dynamiclib -arch arm64 \
        -isysroot "$(xcrun --sdk "${sdk}" --show-sdk-path)" "${min_flag}" \
        -Wl,-force_load,"${dir}/combined.a" \
        -framework Foundation -framework Metal -framework Accelerate \
        -install_name '@rpath/llama.framework/llama' -o "${fw}/llama" \
        >>"${dir}/link.log" 2>&1
    cp "${SOURCE}/include/llama.h" "${SOURCE}/tools/mtmd/mtmd.h" \
        "${SOURCE}/tools/mtmd/mtmd-helper.h" "${fw}/Headers/"
    cp "${SOURCE}/include/llama-cpp.h" "${SOURCE}/src/llama-ext.h" \
        "${SOURCE}/common/common.h" "${SOURCE}/common/speculative.h" \
        "${SOURCE}/common/sampling.h" "${fw}/Headers/"
    local header
    for header in ggml.h ggml-opt.h ggml-alloc.h ggml-backend.h ggml-metal.h ggml-cpu.h ggml-blas.h gguf.h; do
        cp "${SOURCE}/ggml/include/${header}" "${fw}/Headers/"
    done
    cp "${SOURCE}/LICENSE" "${fw}/LICENSE"
    cp "${SOURCE}/vendor/cpp-httplib/LICENSE" "${fw}/LICENSE-cpp-httplib"
    cat >"${fw}/Modules/module.modulemap" <<'MODULE'
framework module llama {
    header "llama.h"
    header "mtmd.h"
    header "mtmd-helper.h"
    header "ggml.h"
    header "ggml-opt.h"
    header "ggml-alloc.h"
    header "ggml-backend.h"
    header "ggml-metal.h"
    header "ggml-cpu.h"
    header "ggml-blas.h"
    header "gguf.h"
    explicit module cpp {
        requires cplusplus
        header "llama-cpp.h"
        header "llama-ext.h"
        header "common.h"
        header "speculative.h"
        header "sampling.h"
        export *
    }
    link "c++"
    link framework "Accelerate"
    link framework "Metal"
    link framework "Foundation"
    export *
}
MODULE
    cat >"${fw}/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>llama</string>
<key>CFBundleIdentifier</key><string>org.ggml.llama.minicpm</string>
<key>CFBundleName</key><string>llama</string>
<key>CFBundlePackageType</key><string>FMWK</string>
<key>CFBundleShortVersionString</key><string>1.0</string>
<key>CFBundleVersion</key><string>1</string>
<key>CFBundleSupportedPlatforms</key><array><string>${platform}</string></array>
<key>MinimumOSVersion</key><string>${minimum}</string>
</dict></plist>
PLIST
    XC_ARGS+=(-framework "${fw}")
    echo "Finished ${name}"
}

if [[ "${MODE}" == all || "${MODE}" == macos ]]; then
    build_slice macos-arm64 macosx MacOSX "${MACOS_MIN}" Darwin
fi
if [[ "${MODE}" == all || "${MODE}" == ios || "${MODE}" == ios-device ]]; then
    build_slice ios-arm64 iphoneos iPhoneOS "${IOS_MIN}" iOS
fi
if [[ "${MODE}" == all || "${MODE}" == ios || "${MODE}" == ios-sim ]]; then
    build_slice ios-simulator-arm64 iphonesimulator iPhoneSimulator "${IOS_MIN}" iOS
fi

# Preserve a working framework until all requested slices have built.
STAGED="${WORK}/llama.xcframework"
rm -rf "${STAGED}"
xcrun xcodebuild -create-xcframework "${XC_ARGS[@]}" -output "${STAGED}"
cat >"${STAGED}/PROVENANCE.txt" <<PROVENANCE
Runtime: https://github.com/tc-mb/llama.cpp-omni
Commit: ${REVISION}
Source archive SHA256: ${ARCHIVE_SHA256}
Local patch: ${LOCAL_PATCH_ID}
Local patch file: scripts/patches/minicpm-metal-worker-autorelease.patch
Local patch SHA256: ${LOCAL_PATCH_SHA256}
Local patch scope: autorelease pools on Metal encoding and residency-refresh workers; no model math changes
Reference: OpenBMB/MiniCPM-V-Apps commit 68f53d67a418dd6444659c39005dd5cc4097071c
Built targets: llama, mtmd, llama-common (and ggml, common-base, cpp-httplib dependencies)
Features: LES expert loading, DSpark speculative decoding
Architectures: arm64; requested platforms: ${MODE}
PROVENANCE
cp "${SOURCE}/LICENSE" "${STAGED}/LICENSE"
cp "${SOURCE}/vendor/cpp-httplib/LICENSE" "${STAGED}/LICENSE-cpp-httplib"
rm -rf "${DEST}"
mv "${STAGED}" "${DEST}"
echo "Installed ${DEST}"
