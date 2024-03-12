# bitlinear-pytorch

Implementation of the BitLinear layer from: [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)

## Install
```bash
pip install bitlinear_pytorch
```

## Usage
```python
import torch
from bitlinear_pytorch import BitLinear, replace_linear_with_bitlinear

class TinyMLP(nn.Module):
    def __init__(self):
        super(TinyMLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.layers(x)

model = TinyMLP()
replace_linear_with_bitlinear(model)

# or use BitLinear directly
bitlinear = BitLinear(784, 256)
```

## License
MIT

## Citation
```bibtex
@misc{ma2024era,
      title={The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits}, 
      author={Shuming Ma and Hongyu Wang and Lingxiao Ma and Lei Wang and Wenhui Wang and Shaohan Huang and Li Dong and Ruiping Wang and Jilong Xue and Furu Wei},
      year={2024},
      eprint={2402.17764},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## TODO
- [x] Implement base BitLinear layer
- [x] Add example usage
- [ ] Implement memory efficient weight encoding/decoding
- [ ] Implement Fast Inference (CUDA/CPU/VHDL)
