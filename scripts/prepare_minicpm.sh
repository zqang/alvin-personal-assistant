#!/usr/bin/env bash
# Build the pinned MiniCPM-V runtime (llama.xcframework) from a fresh extraction
# of the hash-verified source archive, and package it as the release asset that
# MiniCPM/Package.swift pins by checksum.
# Requires Xcode and python3. CMake comes from a hash-pinned PyPI wheel in
# build/cmake-venv (scripts/cmake-requirements.txt) unless CMAKE=/path/to/cmake
# (3.28 or later) is set.
# Usage: scripts/prepare_minicpm.sh [all|macos|ios|ios-device|ios-sim]
# Installs the xcframework into MiniCPM/Frameworks/ (MINICPM_INSTALL=0 skips
# that). `all` also writes build/minicpm/llama.xcframework.zip and
# build/minicpm/PROVENANCE.txt and prints the release tag and checksum.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${ROOT}/build/minicpm"
DEST="${ROOT}/MiniCPM/Frameworks/llama.xcframework"
# Runtime pinned by OpenBMB/MiniCPM-V-Apps at this commit:
APPS_REVISION=68f53d67a418dd6444659c39005dd5cc4097071c
REVISION=96bbf61710d44421ea54e63f5194f9b17c20d877
ARCHIVE_URL="https://codeload.github.com/tc-mb/llama.cpp-omni/tar.gz/${REVISION}"
ARCHIVE_SHA256=5c8c3a490cb7ae216c24a1bdcdddcd1d048b28d90285978d3ca7205c9a7ff8b0
SOURCE="${WORK}/llama.cpp-omni-${REVISION}"
# ponytail: keep this cached archive. GitHub doesn't promise codeload tarballs stay
# byte-identical, so a fresh download of the same commit may fail ARCHIVE_SHA256 (it
# fails closed). Attaching runtime.tar.gz to the runtime release would keep the input.
ARCHIVE="${WORK}/runtime.tar.gz"
# Unproven on the phone: device check D20 decides whether it stays.
LOCAL_PATCH_ID=minicpm-metal-worker-autorelease-v1
LOCAL_PATCH="${ROOT}/scripts/patches/minicpm-metal-worker-autorelease.patch"
MODE="${1:-all}"
IOS_MIN=17.0
MACOS_MIN=14.0
JOBS="${JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || printf 4)}"

case "${MODE}" in all|macos|ios|ios-device|ios-sim) ;; *)
    echo "Usage: $0 [all|macos|ios|ios-device|ios-sim]" >&2; exit 2 ;;
esac
command -v xcrun >/dev/null || { echo 'Xcode is required.' >&2; exit 1; }
if [[ -z "${CMAKE:-}" ]]; then
    VENV="${ROOT}/build/cmake-venv"
    [[ -x "${VENV}/bin/python" ]] || python3 -m venv "${VENV}"
    "${VENV}/bin/python" -m pip install --quiet --disable-pip-version-check \
        --require-hashes --only-binary=:all: -r "${ROOT}/scripts/cmake-requirements.txt"
    CMAKE="${VENV}/bin/cmake"
fi
command -v "${CMAKE}" >/dev/null || {
    echo "CMake not found at ${CMAKE}; unset CMAKE to use the pinned wheel." >&2
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
mkdir -p "${WORK}"

if [[ ! -f "${ARCHIVE}" ]]; then
    curl --fail --location --retry 3 "${ARCHIVE_URL}" -o "${ARCHIVE}.partial"
    mv "${ARCHIVE}.partial" "${ARCHIVE}"
fi
ACTUAL_SHA256="$(shasum -a 256 "${ARCHIVE}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${ARCHIVE_SHA256}" ]]; then
    echo "Runtime archive checksum mismatch: ${ARCHIVE}" >&2
    exit 1
fi

# Pinned upstream plus one explicit ownership-only patch, always on a fresh
# extraction, so no earlier edit or partial build can reach a release.
rm -rf "${SOURCE}"
tar -xzf "${ARCHIVE}" -C "${WORK}"
patch -f -s -F 0 -p1 -d "${SOURCE}" -i "${LOCAL_PATCH}" || {
    echo "${LOCAL_PATCH_ID} does not apply to the pinned runtime; refusing to build." >&2
    exit 1
}
LOCAL_PATCH_SHA256="$(shasum -a 256 "${LOCAL_PATCH}" | awk '{print $1}')"

# Notices shipped in every llama.framework: the runtime, its licenses/ folder
# (nlohmann/json and cpp-httplib among them), and the licences of the code
# compiled in from single headers and source comments. A missing one fails the
# build.
NOTICES="${WORK}/LICENSES"
rm -rf "${NOTICES}"
mkdir -p "${NOTICES}"
cp "${SOURCE}/LICENSE" "${NOTICES}/LICENSE-llama.cpp"
cp "${SOURCE}"/licenses/* "${NOTICES}/"
header_notice() { # name, header, first line of its licence comment
    sed -n "/$3/,/^\*\//p" "${SOURCE}/$2" >"${NOTICES}/LICENSE-$1"
}
header_notice stb_image vendor/stb/stb_image.h '^This software is available under 2 licenses'
header_notice miniaudio vendor/miniaudio/miniaudio.h '^This software is available as a choice'
# llamafile's sgemm (GGML_LLAMAFILE is on by default) carries the MIT text in its header.
sed -n '1,/^\/\/ SOFTWARE\.$/p' "${SOURCE}/ggml/src/ggml-cpu/llamafile/sgemm.cpp" \
    | sed 's|^// \{0,1\}||' >"${NOTICES}/LICENSE-llamafile-sgemm"
mit_notice() { # name, file, its copyright line: that line, then the MIT terms
    grep -m 1 -o "$3" "${SOURCE}/$2" >"${NOTICES}/LICENSE-$1" || true
    { echo; sed -n '/^Permission is hereby granted/,$p' "${SOURCE}/LICENSE"; } >>"${NOTICES}/LICENSE-$1"
}
# These name MIT and their copyright holder but don't carry the text.
mit_notice minja vendor/minja/minja.hpp 'Copyright 2024 Google LLC'
mit_notice yarn ggml/src/ggml-cpu/ops.cpp 'Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng'
for notice in llama.cpp jsonhpp httplib minja stb_image miniaudio llamafile-sgemm yarn; do
    { grep -q Copyright "${NOTICES}/LICENSE-${notice}" &&
        grep -q 'Permission is hereby granted' "${NOTICES}/LICENSE-${notice}"; } 2>/dev/null || {
        echo "Licence notice missing or incomplete: LICENSE-${notice}" >&2; exit 1
    }
done

# The framework's required-reason APIs; build_slice checks them against nm -u.
PRIVACY="${WORK}/PrivacyInfo.xcprivacy"
cat >"${PRIVACY}" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>NSPrivacyTracking</key><false/>
<key>NSPrivacyTrackingDomains</key><array/>
<key>NSPrivacyCollectedDataTypes</key><array/>
<key>NSPrivacyAccessedAPITypes</key><array>
<dict>
<key>NSPrivacyAccessedAPIType</key><string>NSPrivacyAccessedAPICategoryFileTimestamp</string>
<key>NSPrivacyAccessedAPITypeReasons</key><array><string>C617.1</string></array>
</dict>
</array>
</dict></plist>
PLIST
plutil -lint "${PRIVACY}" >/dev/null

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
SLICES=()

build_slice() {
    local name="$1" sdk="$2" platform="$3" minimum="$4" system="$5"
    local dir="${WORK}/${name}" release=Release
    local fw="${WORK}/${name}/framework/llama.framework"
    local extra_args=() min_flag reasons
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
    rm -rf "${dir}"
    mkdir -p "${dir}"
    echo "Configuring ${name}; logs: ${dir}/configure.log"
    # The +-expansion keeps an empty array legal under set -u in macOS's bash 3.2.
    if ! "${CMAKE}" -S "${SOURCE}" -B "${dir}" "${COMMON_ARGS[@]}" \
        ${extra_args[@]+"${extra_args[@]}"} -DCMAKE_OSX_SYSROOT="${sdk}" \
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

    # Required-reason APIs the binary imports must match PrivacyInfo.xcprivacy:
    # stat is the file-timestamp category (C617.1).
    # ponytail: nm sees C symbols only; an Objective-C selector such as
    # systemUptime would need a strings scan if a runtime update adds one.
    reasons="$(xcrun nm -u "${fw}/llama" | grep -xE \
        '_(f?stat(at)?|lstat|f?getattrlist(at|bulk)?|f?statv?fs|mach_absolute_time)|_OBJC_CLASS_[$]_NSUserDefaults' \
        | sort | xargs)" || true
    if [[ "${reasons}" != _stat ]]; then
        echo "${name} imports required-reason APIs '${reasons}'; update PrivacyInfo.xcprivacy." >&2
        exit 1
    fi
    cp -R "${NOTICES}" "${fw}/LICENSES"
    cp "${PRIVACY}" "${fw}/PrivacyInfo.xcprivacy"

    cp "${SOURCE}/include/llama.h" "${SOURCE}/tools/mtmd/mtmd.h" \
        "${SOURCE}/tools/mtmd/mtmd-helper.h" "${fw}/Headers/"
    cp "${SOURCE}/include/llama-cpp.h" "${SOURCE}/src/llama-ext.h" \
        "${SOURCE}/common/common.h" "${SOURCE}/common/speculative.h" \
        "${SOURCE}/common/sampling.h" "${fw}/Headers/"
    local header
    for header in ggml.h ggml-opt.h ggml-alloc.h ggml-backend.h ggml-metal.h ggml-cpu.h ggml-blas.h gguf.h; do
        cp "${SOURCE}/ggml/include/${header}" "${fw}/Headers/"
    done
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
    SLICES+=("${name}: $(xcrun lipo -archs "${fw}/llama"), ${sdk} SDK $(xcrun --sdk "${sdk}" --show-sdk-version), minimum ${minimum}, llama SHA256 $(shasum -a 256 "${fw}/llama" | awk '{print $1}')")
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
# Anything that changes the asset changes the tag (never re-upload under an old
# one: older commits pin its checksum): the patch and this script are hashed in.
SCRIPT_SHA256="$(shasum -a 256 "${BASH_SOURCE[0]}" | awk '{print $1}')"
TAG="minicpm-runtime-${REVISION:0:7}-p${LOCAL_PATCH_SHA256:0:8}-s${SCRIPT_SHA256:0:8}"
rm -rf "${STAGED}" "${WORK}/llama.xcframework.zip" "${WORK}/PROVENANCE.txt"
xcrun xcodebuild -create-xcframework "${XC_ARGS[@]}" -output "${STAGED}"
{
    echo "Runtime: https://github.com/tc-mb/llama.cpp-omni commit ${REVISION}"
    echo "Pinned by: OpenBMB/MiniCPM-V-Apps commit ${APPS_REVISION}"
    echo "Source archive: ${ARCHIVE_URL}"
    echo "Source archive SHA256: ${ARCHIVE_SHA256} (verified, extracted fresh)"
    echo "Local patch: ${LOCAL_PATCH_ID}, scripts/patches/$(basename "${LOCAL_PATCH}") SHA256 ${LOCAL_PATCH_SHA256}"
    echo "Release tag: ${TAG}"
    echo "Build script: scripts/prepare_minicpm.sh SHA256 ${SCRIPT_SHA256}"
    echo "Repository: $(git -C "${ROOT}" describe --always --dirty 2>/dev/null || echo unknown)"
    echo "Built: $(date -u +%Y-%m-%dT%H:%M:%SZ) on macOS $(sw_vers -productVersion) $(uname -m)"
    echo "Toolchain: $(xcrun xcodebuild -version | paste -sd ' ' -), CMake ${CMAKE_VERSION}"
    echo "CMake targets: llama mtmd llama-common; arguments: ${COMMON_ARGS[*]}"
    echo "Requested platforms: ${MODE}"
    printf 'Slice %s\n' "${SLICES[@]}"
    echo "Notices in each llama.framework/LICENSES: $(cd "${NOTICES}" && echo *)"
    echo "Required-reason APIs: stat (NSPrivacyAccessedAPICategoryFileTimestamp, C617.1)"
} >"${STAGED}/PROVENANCE.txt"

# Only a three-slice build becomes a release asset.
if [[ "${MODE}" == all ]]; then
    ZIP="${WORK}/llama.xcframework.zip"
    ditto -c -k --norsrc --noextattr --noacl --keepParent "${STAGED}" "${ZIP}"
    CHECKSUM="$(swift package compute-checksum "${ZIP}")"
    { cat "${STAGED}/PROVENANCE.txt"
      echo "Release asset: llama.xcframework.zip, swift package compute-checksum ${CHECKSUM}"
    } >"${WORK}/PROVENANCE.txt"
    echo "Release asset: ${ZIP}"
    echo "Release tag: ${TAG}"
    echo "Checksum: ${CHECKSUM}"
fi
if [[ "${MINICPM_INSTALL:-1}" != 0 ]]; then
    rm -rf "${DEST}"
    mkdir -p "$(dirname "${DEST}")"
    ditto "${STAGED}" "${DEST}"
    echo "Installed ${DEST}"
fi
