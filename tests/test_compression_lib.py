import pytest
import torch
from compression_lib import CompressionLib, create_compression_lib

class TestCompressionLib:
    def test_init(self):
        m = CompressionLib(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = CompressionLib(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_compression_lib()
        assert isinstance(m, CompressionLib)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
