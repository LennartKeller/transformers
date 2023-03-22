# Notes

## Build Faiss on Apple Silicon

```bash
brew install llvm swig
```

```bash
LDFLAGS="-L/opt/homebrew/opt/llvm/lib" CPPFLAGS="-I/opt/homebrew/opt/llvm/include" CXX=/opt/homebrew/opt/llvm/bin/clang++ CC=/opt/homebrew/opt/llvm/bin/clang cmake -DFAISS_ENABLE_GPU=OFF -B build .
```

Then follow the instructions from the repo!