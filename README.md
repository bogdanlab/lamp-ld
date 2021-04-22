# LAMP-LD

## Install python version
```bash
pip install -e .
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
