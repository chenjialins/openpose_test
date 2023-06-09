import torch
import torch.nn as nn
from torchvision import models

# gpu device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

class ConvLstm(nn.Module):
    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional, n_class):
        super(ConvLstm, self).__init__()
        self.conv_model = Pretrained_conv(latent_dim)
        self.Lstm = Lstm(latent_dim, hidden_size, lstm_layers, bidirectional)
        self.output_layer = nn.Sequential(
            nn.Linear(2 * hidden_size if bidirectional==True else hidden_size, n_class),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        batch_size, timesteps, channel_x, h_x, w_x = x.shape
        conv_input = x.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.conv_model(conv_input)
        lstm_input = conv_output.view(batch_size, timesteps, -1)
        lstm_output = self.Lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]
        output = self.output_layer(lstm_output)
        return output

class Pretrained_conv(nn.Module):
    def __init__(self, latent_dim):
        super(Pretrained_conv, self).__init__()
        self.conv_model = models.resnet18(pretrained=True)
        # ====== freezing all of the layers ======
        for param in self.conv_model.parameters():
            param.requires_grad = False
        # ====== changing the last FC layer to an output with the size we need. this layer is un freezed ======
        self.conv_model.fc = nn.Linear(self.conv_model.fc.in_features, latent_dim)

    def forward(self, x):
        return self.conv_model(x)

class Lstm(nn.Module):
    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional):
        super(Lstm, self).__init__()
        self.Lstm = nn.LSTM(latent_dim, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self,x):
        output, self.hidden_state = self.Lstm(x, self.hidden_state)
        return output


if __name__ == "__main__":
    x = torch.randn([20, 10, 3, 224, 224])
    net = ConvLstm(512, 256, 2, True, 2)
    net = net.to(device)
    x = x.to(device)
    out = net(x)
    print(out.shape)
    # def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional, n_class):
    # parser.add_argument('--latent_dim', default=512, type=int, help='The dim of the Conv FC output (default:512)')
    # parser.add_argument('--hidden_size', default=256, type=int,
    #                     help="The number of features in the LSTM hidden state (default:256)")
    # parser.add_argument('--lstm_layers', default=2, type=int, help='Number of recurrent layers (default:2)')
    # parser.add_argument('--bidirectional', default=True, type=bool, help='set the LSTM to be bidirectional (default:True)')
    