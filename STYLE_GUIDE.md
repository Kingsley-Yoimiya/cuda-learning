#### C++
C++ code should conform to [LLVM Style Guide]([LLVM Coding Standards (apple.com)](https://opensource.apple.com/source/lldb/lldb-112/llvm/docs/CodingStandards.html)).

Addons uses [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
to check your C/C++ changes. Sometimes you have some manually formatted
code that you don’t want clang-format to touch.
You can disable formatting like this:

```cpp
int formatted_code;
// clang-format off
    void    unformatted_code  ;
// clang-format on
void formatted_code_again;
```

Install Clang-format 9 for Ubuntu:

```bash
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add - 
sudo add-apt-repository -u 'http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main'
sudo apt install clang-format-9
```

format all with:
```bash
clang-format-9 -i --style=LLVM simple_practice/*/*.cu
```

Install Clang-format for MacOS:
```bash
brew update
brew install clang-format
```

format all with:
```bash
clang-format -i --style=LLVM simple_practice/*/*.cu
```
