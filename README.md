# bitlinear-pytorch

Implementation of the BitLinear layer from: [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)

## Install
```bash
pip install bitlinear-pytorch
```

## Usage
```python
from torch import nn
from bitlinear_pytorch import BitLinear

class TinyMLP(nn.Module):
    def __init__(self):
        super(TinyMLP, self).__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(784),
            BitLinear(784, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            BitLinear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            BitLinear(128, 10),
        )

    def forward(self, x):
        return self.layers(x)

model = TinyMLP()
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
- [x] Setup Github Actions workflow
- [ ] Implement memory efficient weight encoding/decoding
- [ ] Implement Fast Inference (CUDA/CPU/VHDL)
