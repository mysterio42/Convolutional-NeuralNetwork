import torch.nn as nn


class Convolutional(nn.Module):

    def __init__(self):
        super(Convolutional, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        self.avgpooling_1 = nn.AvgPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        self.avgpooling_2 = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, x):
        def fullyconnected_layer():
            def avgpooling_layer_2():
                def conv_layer_2():
                    def avgpooling_layer_1():
                        def conv_layer_1():
                            return self.relu1(self.cnn1(x))

                        return self.avgpooling_1(conv_layer_1())

                    return self.relu2(self.cnn2(avgpooling_layer_1()))

                return self.avgpooling_2(conv_layer_2())

            out = avgpooling_layer_2()
            out = out.view(out.size(0), -1)
            return self.fc1(out)

        return fullyconnected_layer()
