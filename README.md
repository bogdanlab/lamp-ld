# LAMP-LD
![python package](https://github.com/KangchengHou/lamp-ld/actions/workflows/python.yml/badge.svg)
## Install python version
```bash
pip install git+https://github.com/bogdanlab/lamp-ld.git#egg=pylampld
```

If you see errors, we recommend the following:
```bash
mamba install gcc_linux-64 g++_linux-64
pip install cmake
```
## Install executable version

```bash
mkdir build && cd build
cmake .. -DBUILD_EXE=True
make
```

## Example usage
```bash
lampld --pos pos.txt --admix admix.hap --ref EUR.hap AFR.hap EAS.hap --out out.txt
```

## Contact
Kangcheng Hou (kangchenghou@gmail.com)
