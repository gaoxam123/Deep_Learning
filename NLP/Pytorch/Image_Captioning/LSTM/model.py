import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super().__init__()
        self.train_CNN = train_CNN
        self.embed_size = embed_size
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        features = self.inception(x)

        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True

            else:
                param.requires_grad = self.train_CNN

        return self.dropout(self.relu(features))
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, source, target):
        embeddings = self.dropout(self.word_embeddings(target))
        embeddings = torch.cat([source.unsqueeze(0), embeddings], dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.fc(hiddens)
        
        return outputs
    
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, x, captions):
        features = self.encoderCNN(x)
        outputs = self.decoderRNN(features, captions)

        return outputs
    
    def caption_image(self, images, vocab, max_len=50):
        result = []

        with torch.no_grad():
            x = self.encoderCNN(images).unsqueeze(0)
            states = None

            for _ in range(max_len):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.fc(hiddens.unsqueeze(0))
                predicted = output.argmax(1)

                result.append(predicted.item())
                x = self.decoderRNN.word_embeddings(predicted).unsqueeze(0)

                if vocab.itos[predicted.item()] == "<EOS>":
                    break

            return [vocab.itos[idx] for idx in result]