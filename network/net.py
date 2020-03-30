import torch.nn as nn


class Convolutional(nn.Module):

    def __init__(self):
        super(Convolutional, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        self.maxpooling_1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        self.maxpooling_2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=32 * 4 * 4, out_features=10)

    def forward(self, x):
        def fullyconnected_layer():
            def maxpooling_layer_2():
                def conv_layer_2():
                    def maxpooling_layer_1():
                        def conv_layer_1():
                            return self.relu1(self.cnn1(x))

                        return self.maxpooling_1(conv_layer_1())

                    return self.relu2(self.cnn2(maxpooling_layer_1()))

                return self.maxpooling_2(conv_layer_2())

            out = maxpooling_layer_2()
            out = out.view(out.size(0), -1)
            return self.fc1(out)

        return fullyconnected_layer()
