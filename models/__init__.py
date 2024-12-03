#from .cond_cnnDG import cCNNDiscriminator, cCNNGenerator
from .conv32 import CNNDiscriminator, CNNGenerator
from .conv256 import CNNDiscriminator256, CNNGenerator256
from .resnet32 import ResGenerator32, ResDiscriminator32
from .resnet256 import ResNetGenerator, ResNetDiscriminator
from .get_model import get_model

__all__ = ['CNNDiscriminator', 'CNNGenerator','ResNetGenerator','CNNGenerator256','ResNetGenerator256',
           'ResNetDiscriminator','get_model']