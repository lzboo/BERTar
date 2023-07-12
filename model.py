import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import fm

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Load RNA-FM model
def get_embed(batch_tokens):
    fm_model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fm_model = fm_model.to(device)
    fm_model.eval()  # disables dropout for deterministic results

    # batch_tokens =batch_tokens.squeeze(0)
    with torch.no_grad():
        batch_tokens = batch_tokens.to(device)
        results = fm_model(batch_tokens, repr_layers=[1])
    token_embeddings = results["representations"][1]
    return token_embeddings
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class HyperParam:
    def __init__(self, filters=None, kernels=None, model_json=None):
        self.dictionary = dict()
        self.name_postfix = str()

        if (filters is not None) and (kernels is not None) and (model_json is None):
            for i, (f, k) in enumerate(zip(filters, kernels)):
                setattr(self, 'f{}'.format(i+1), f)
                setattr(self, 'k{}'.format(i+1), k)
                self.dictionary.update({'f{}'.format(i+1): f, 'k{}'.format(i+1): k})
            self.len = i+1
                
            for key, value in self.dictionary.items():
                self.name_postfix = "{}_{}-{}".format(self.name_postfix, key, value)
        elif model_json is not None:
            self.dictionary = json.loads(model_json)
            for i, (key, value) in enumerate(self.dictionary.items()):
                setattr(self, key, value)
                self.name_postfix = "{}_{}-{}".format(self.name_postfix, key, value)
            self.len = (i+1)//2
    
    def __len__(self):
        return self.len

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class BERTarI_FC(nn.Module):
    def __init__(self, hparams=None, hidden_units=1000, input_shape=(2, 30), name_prefix="model"):
        super(BERTarI_FC, self).__init__()

        self.fc_mi = nn.Linear(640, 32)
        self.fc_mr = nn.Linear(640, 32)
        """ out_features = ((in_length - kernel_size + (2 * padding)) / stride + 1) * out_channels """
        # flat_features = self.forward(torch.randint(1, 5, input_shape).to(device), torch.randint(1, 5, input_shape).to(device), flat_check=False)
        self.fc1 = nn.Linear(1920, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 2)


    def forward(self, x_mirna, x_mrna, flat_check=False):
        mi_out = get_embed(x_mirna)
        mi_out = self.fc_mi(mi_out).transpose(1,2)
        mr_out = get_embed(x_mrna)
        mr_out = self.fc_mr(mr_out).transpose(1,2)
        # import pdb;pdb.set_trace()
        h_mirna = F.relu(mi_out)                   # torch.Size([32, 32, 30])
        # print(h_mirna.shape)
        h_mrna = F.relu(mr_out)                    # torch.Size([32, 32, 30])
        # print(h_mrna.shape)
        h = torch.cat((h_mirna, h_mrna), dim=1)    # torch.Size([32, 64, 30])
        # print(h.shape)

        h = h.view(h.size(0), -1)                  # torch.Size([32, 1920])
        # print(h.shape)
        if flat_check:
            return h.size(1)
        h = self.fc1(h)                            # torch.Size([32, 1000])
        # print(h.shape)
        # y = F.softmax(self.fc2(h), dim=1)
        y = self.fc2(h)                            # torch.Size([32, 2])
        # print(y.shape)

        return y



class BERTarI_CNN(nn.Module):
    def __init__(self, hparams=None, hidden_units=1000, input_shape=(2, 30), name_prefix="model"):
        super(BERTarI_CNN, self).__init__()

        if hparams is None:
            filters, kernels = [32, 16, 64, 16], [3, 3, 3, 3]
            hparams = HyperParam(filters, kernels)
        self.name = "{}{}".format(name_prefix, hparams.name_postfix)
        self.fc_mi = nn.Linear(640, 32)
        self.fc_mr = nn.Linear(640, 32)

        if (isinstance(hparams, HyperParam)) and (len(hparams) == 4):
            self.embd1 = nn.Conv1d(4, hparams.f1, kernel_size=hparams.k1, padding=((hparams.k1 - 1) // 2))

            self.conv2 = nn.Conv1d(hparams.f1*2, hparams.f2, kernel_size=hparams.k2)
            self.conv3 = nn.Conv1d(hparams.f2, hparams.f3, kernel_size=hparams.k3)
            self.conv4 = nn.Conv1d(hparams.f3, hparams.f4, kernel_size=hparams.k4)

            """ out_features = ((in_length - kernel_size + (2 * padding)) / stride + 1) * out_channels """
            # flat_features = self.forward(torch.randint(1, 5, input_shape).to(device), torch.randint(1, 5, input_shape).to(device), flat_check=False)
            self.fc1 = nn.Linear(384, hidden_units)
            self.fc2 = nn.Linear(hidden_units, 2)
        else:
            raise ValueError("not enough hyperparameters")

    def forward(self, x_mirna, x_mrna, flat_check=False):
        mi_out = get_embed(x_mirna)
        mi_out = self.fc_mi(mi_out).transpose(1,2)
        mr_out = get_embed(x_mrna)
        mr_out = self.fc_mr(mr_out).transpose(1,2)
        # import pdb;pdb.set_trace()
        h_mirna = F.relu(mi_out)                   # torch.Size([32, 32, 30])
        # print(h_mirna.shape)
        h_mrna = F.relu(mr_out)                    # torch.Size([32, 32, 30])
        # print(h_mrna.shape)
        h = torch.cat((h_mirna, h_mrna), dim=1)    # torch.Size([32, 64, 30])
        # print(h.shape)
        h = F.relu(self.conv2(h))                  # torch.Size([32, 16, 28])
        # print(h.shape)
        h = F.relu(self.conv3(h))                  # torch.Size([32, 64, 26])
        # print(h.shape)
        h = F.relu(self.conv4(h))                  # torch.Size([32, 16, 24])
        # print(h.shape)

        h = h.view(h.size(0), -1)                  # torch.Size([32, 384])
        # print(h.shape)
        if flat_check:
            return h.size(1)
        h = self.fc1(h)                            # torch.Size([32, 1000])
        # print(h.shape)
        # y = F.softmax(self.fc2(h), dim=1)
        y = self.fc2(h)                            # torch.Size([32, 2])
        # print(y.shape)

        return y

    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



class BERTarI_lstm(nn.Module):
    def __init__(self):
        super(BERTarI_lstm, self).__init__()

        output_size = 2
        n_layers = 5
        hidden_dim = 15
        drop_prob=0.5

        self.fc_mi = nn.Linear(640, 32)
        self.fc_mr = nn.Linear(640, 32)

        '''
        #Bert ----------------重点，bert模型需要嵌入到自定义模型里面
        self.bert=BertModel.from_pretrained(bertpath)
        for param in self.bert.parameters():
            param.requires_grad = True
        '''

        # LSTM layers
        self.lstm = nn.LSTM(30, hidden_dim, n_layers, batch_first=True, bidirectional=True)

        # dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # linear and sigmoid layers

        self.fc = nn.Linear(hidden_dim*2, output_size)

        #self.sig = nn.Sigmoid()



    def forward(self,  x_mirna, x_mrna, flat_check=False):

        mi_out = get_embed(x_mirna)
        mi_out = self.fc_mi(mi_out).transpose(1,2)
        mr_out = get_embed(x_mrna)
        mr_out = self.fc_mr(mr_out).transpose(1,2)
        # import pdb;pdb.set_trace()
        h_mirna = F.relu(mi_out)                   # torch.Size([32, 32, 30])
        # print(h_mirna.shape)
        h_mrna = F.relu(mr_out)                    # torch.Size([32, 32, 30])
        # print(h_mrna.shape)
        h = torch.cat((h_mirna, h_mrna), dim=1)    # torch.Size([32, 64, 30])

        # lstm_out
        #x = x.float()
        lstm_out, (hidden_last,cn_last) = self.lstm(h)
        #print(lstm_out.shape)   #[32,100,768]
        #print(hidden_last.shape)   #[4, 32, 384]
        #print(cn_last.shape)    #[4, 32, 384]

        #修改 双向的需要单独处理

        #正向最后一层，最后一个时刻
        hidden_last_L=hidden_last[-2]
        #print(hidden_last_L.shape)  #[32, 384]
        #反向最后一层，最后一个时刻
        hidden_last_R=hidden_last[-1]
        #print(hidden_last_R.shape)   #[32, 384]
        #进行拼接
        hidden_last_out=torch.cat([hidden_last_L,hidden_last_R],dim=-1)



        # dropout and fully-connected layer
        out = self.dropout(hidden_last_out)
        #print(out.shape)    #[32,768]
        out = self.fc(out)

        return out

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class BertarII_FC(nn.Module):
    def __init__(self, hparams=None, hidden_units=1000, input_shape=(2, 30), name_prefix="model"):
        super(BertarII_FC, self).__init__()

        self.fc_mi = nn.Linear(640, 32)
        # self.fc_mr = nn.Linear(640, 32)
        """ out_features = ((in_length - kernel_size + (2 * padding)) / stride + 1) * out_channels """
        # flat_features = self.forward(torch.randint(1, 5, input_shape).to(device), torch.randint(1, 5, input_shape).to(device), flat_check=False)
        self.fc1 = nn.Linear(1920, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 2)


    def forward(self, rna, flat_check=False):
        mi_out = get_embed(rna)
        mi_out = self.fc_mi(mi_out).transpose(1,2)
        # mr_out = get_embed(x_mrna)
        # mr_out = self.fc_mr(mr_out).transpose(1,2)
        # import pdb;pdb.set_trace()
        h = F.relu(mi_out)                   # torch.Size([32, 32, 30])
        # print(h.shape)                       # torch.Size([32, 32, 30])
        # h_mrna = F.relu(mr_out)                    # torch.Size([32, 32, 30])
        # print(h_mrna.shape)
        # h = torch.cat((h_mirna, h_mrna), dim=1)    # torch.Size([32, 64, 30])
        # print(h.shape)

        h = h.reshape(h.size(0), -1)                  # torch.Size([32, 1920])
        # print(h.shape)
        if flat_check:
            return h.size(1)
        h = self.fc1(h)                            # torch.Size([32, 1000])
        # print(h.shape)
        # y = F.softmax(self.fc2(h), dim=1)
        y = self.fc2(h)                            # torch.Size([32, 2])
        # print(y.shape)

        return y


class BertarII_CNN(nn.Module):
    def __init__(self, hparams=None, hidden_units=1000, input_shape=(2, 30), name_prefix="model"):
        super(BertarII_CNN, self).__init__()

        if hparams is None:
            filters, kernels = [32, 16, 64, 16], [3, 3, 3, 3]
            hparams = HyperParam(filters, kernels)
        self.name = "{}{}".format(name_prefix, hparams.name_postfix)
        self.fc_mi = nn.Linear(640, 32)
        self.fc_mr = nn.Linear(640, 32)

        if (isinstance(hparams, HyperParam)) and (len(hparams) == 4):
            self.embd1 = nn.Conv1d(4, hparams.f1, kernel_size=hparams.k1, padding=((hparams.k1 - 1) // 2))

            self.conv2 = nn.Conv1d(hparams.f1, hparams.f2, kernel_size=hparams.k2)
            self.conv3 = nn.Conv1d(hparams.f2, hparams.f3, kernel_size=hparams.k3)
            self.conv4 = nn.Conv1d(hparams.f3, hparams.f4, kernel_size=hparams.k4)

            """ out_features = ((in_length - kernel_size + (2 * padding)) / stride + 1) * out_channels """
            # flat_features = self.forward(torch.randint(1, 5, input_shape).to(device), torch.randint(1, 5, input_shape).to(device), flat_check=False)
            self.fc1 = nn.Linear(864, hidden_units)
            self.fc2 = nn.Linear(hidden_units, 2)
        else:
            raise ValueError("not enough hyperparameters")

    def forward(self, rna, flat_check=False):
        mi_out = get_embed(rna)
        mi_out = self.fc_mi(mi_out).transpose(1,2)
        h = F.relu(mi_out)                   # torch.Size([128, 32, 60])
        #print(h.shape)
        h = F.relu(self.conv2(h))                  # torch.Size([128, 16, 58])
        #print(h.shape)
        h = F.relu(self.conv3(h))                  # torch.Size([128, 64, 56])
        #print(h.shape)
        h = F.relu(self.conv4(h))                  # torch.Size([128, 16, 54])
        #print(h.shape)

        h = h.reshape(h.size(0), -1)                  # torch.Size([128, 864])
        #print(h.shape)
        if flat_check:
            return h.size(1)
        h = self.fc1(h)                            # torch.Size([32, 1000])
        #print(h.shape)
        # y = F.softmax(self.fc2(h), dim=1)
        y = self.fc2(h)                            # torch.Size([32, 2])
        # print(y.shape)

        return y

    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)