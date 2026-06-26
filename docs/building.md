# Building (step by step)

## 1. Prerequisites
- Python 3.10+
- A C compiler with OpenMP (`clang` + `libomp`)
- Node (only to re-validate the README's mermaid)

```bash
brew install libomp        # OpenMP for clang on macOS
```

## 2. Clone + Python deps
```bash
git clone https://github.com/cluster2600/cluster_compilot.git
cd cluster_compilot
pip install -r requirements.txt        # islpy, certifi
```

## 3. LLM key (for live runs)
Either set an env var, or store it in OpenBao:
```bash
export GEMINI_API_KEY=...               # option A
# option B: OpenBao at secrets/google, field api_key (auto-read; OpenBao must be unsealed)
```

## 4. Smoke test (no key needed)
```bash
python3 -m tests.test_legality          # expect 10/10
python3 run_agent.py --mock             # full agent loop, scripted
```

## 5. (Optional) Build the exact Tiramisu backend
Only needed for real-compiler parity. Builds LLVM 14 + Halide + ISL + `libtiramisu` from source — long, multi-GB.

```bash
cd third_party
git clone --recursive https://github.com/Tiramisu-Compiler/tiramisu.git
cd tiramisu && git checkout 041afad
git submodule update --init --recursive

# Build LLVM/Halide/ISL. If aux LLVM tools (sancov/obj2yaml) fail to compile,
# resume the LLVM build with `ninja -k 0` — the libraries + clang we need still build.
./utils/scripts/install_submodules.sh "$PWD"

# Install Halide so its CMake config (incl. HalideHelpers) is laid out:
cmake --install 3rdParty/Halide/build --prefix "$PWD/3rdParty/Halide/install"

# Configure + build libtiramisu:
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS=-std=c++17 \
  -DUSE_MPI=FALSE -DUSE_GPU=FALSE -DUSE_FLEXNLP=FALSE -DUSE_MKL_WRAPPER=FALSE \
  -DCMAKE_PREFIX_PATH="$PWD/../3rdParty/Halide/install" \
  -DHalide_DIR="$PWD/../3rdParty/Halide/install/lib/cmake/Halide" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make -j tiramisu                        # produces build/libtiramisu.dylib

# Verify ISL vs the real compiler:
cd ../../..
python3 -m tests.test_tiramisu_parity   # expect 4/4
```

> Tiramisu's bundled LLVM is old; on a modern toolchain the auxiliary tools may not compile, but the libraries + Clang do (hence `ninja -k 0`). `CMAKE_POLICY_VERSION_MINIMUM=3.5` is needed because a vendored pybind11 declares a `cmake_minimum_required` newer CMake rejects.
