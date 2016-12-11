from train_cnn import train_cnn
from cnn_3d import Config

def compare_gating():
    config = Config()
    config.downsample_factor = 6
    config.num_layers = 0
    config.num_layers_to_train = 0
    config.mode = 'supervised'
    config.num_layers_to_restore = 0
    config.use_correlation = 0
    config.sum_dir = 'gating_comparison'

    config.use_sex_labels = False
    config.gate = 'male'
    train_cnn(config)

    config.use_sex_labels = True
    config.gate = 'equal_gender'
    train_cnn(config)

    config.use_sex_labels = False
    config.gate = 'shuffle'
    train_cnn(config)

def compare_layers():
    config = Config()
    config.downsample_factor = 6
    config.num_layers = 0
    config.num_layers_to_train = 0
    config.mode = 'supervised'
    config.num_layers_to_restore = 0
    config.use_correlation = 0
    config.sum_dir = 'layer_comparison'
    config.use_sex_labels = False
    config.gate = 'male'
    train_cnn(config)

    config.downsample_factor = 4
    config.num_layers = 2
    config.num_layers_to_train = 2
    train_cnn(config)

    config.downsample_factor = 2
    config.num_layers = 4
    config.num_layers_to_train = 4
    train_cnn(config)

    config.downsample_factor = 0
    config.num_layers = 6
    config.num_layers_to_train = 6
    train_cnn(config)

def compare_data_augmentation_and_pretraining():
    config = Config()
    config.downsample_factor = 0
    config.num_layers = 6
    config.num_layers_to_train = 6
    config.mode = 'supervised'
    config.num_layers_to_restore = 0
    config.use_correlation = 0
    config.sum_dir = 'data_augmentation_comparison'
    config.use_sex_labels = False
    config.gate = 'male'

    config.num_layers_to_restore = 6
    config.num_layers_to_train = 0
    config.rotate = True
    config.noise = 0.1
    train_cnn(config)

    config.num_layers_to_train = 6
    config.rotate = True
    config.noise = 0.1
    train_cnn(config)

    config.num_layers_to_restore = 0
    config.num_layers_to_train = 6
    config.rotate = True
    config.noise = 0.1
    train_cnn(config)

    config.rotate = False
    config.noise = 0
    train_cnn(config)

    config.rotate = False
    config.noise = 1
    train_cnn(config)

def compare_correlation():
    config = Config()
    config.use_correlation = 2
    config.gate = 'male'
    config.sum_dir = 'correlation_comparison'
    config.rotate = True
    config.noise = 0.1
    config.mode = 'supervised'
    train_cnn(config)

compare_correlation()
