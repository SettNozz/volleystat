import torch
import unittest
from src.models.unet import UNet


class TestUNet(unittest.TestCase):
    """Test cases for UNet model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = UNet()
        self.batch_size = 2
        self.input_channels = 6
        self.height = 256
        self.width = 256
        
    def test_model_output_shape(self):
        """Test that the model outputs the correct shape."""
        x = torch.randn(self.batch_size, self.input_channels, self.height, self.width)
        output = self.model(x)
        
        expected_shape = (self.batch_size, 1, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)
        
    def test_model_output_range(self):
        """Test that the model output is in the correct range [0, 1]."""
        x = torch.randn(self.batch_size, self.input_channels, self.height, self.width)
        output = self.model(x)
        
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))


if __name__ == '__main__':
    unittest.main() 