import torch
import torch.nn as nn
import torch.nn.functional as F


class PrefixGenerator(nn.Module):
    def __init__(self, prompt_length=10, layer_num=12, embed_length=197, embed_dim=768, head_num=12,
                 prompt_init='uniform', hyper_mid_dim=1024, mid_dim=16, batch_size=128, layer_embed=False):
        super().__init__()

        self.prompt_length = prompt_length
        self.layer_num = layer_num
        self.head_num = head_num
        self.embed_length = embed_length
        self.hyper_embed_dim = embed_dim
        self.embed_dim = embed_dim // self.head_num
        self.hyper_mid_dim = hyper_mid_dim
        self.mid_dim = mid_dim
        self.prompt_init = prompt_init.lower()
        self.batch_size = batch_size * 2
        self.layer_embed = layer_embed

        # Layer norm
        self.prefix_layer_norm = nn.LayerNorm(self.embed_dim)

        if self.layer_embed:
            self.key_down_prefix_hypernet = PrefixHyperNetLayerEmbedVer(embed_length=self.embed_length,
                                                                        embed_dim=self.hyper_embed_dim,
                                                                        mid_dim=self.hyper_mid_dim,
                                                                        target_mid_dim=self.mid_dim,
                                                                        output_dim=self.embed_dim,
                                                                        num_layers=self.layer_num)

            self.key_up_prefix_hypernet = PrefixHyperNetLayerEmbedVer(embed_length=self.embed_length,
                                                                      embed_dim=self.hyper_embed_dim,
                                                                      mid_dim=self.hyper_mid_dim,
                                                                      target_mid_dim=self.embed_dim,
                                                                      output_dim=self.mid_dim,
                                                                      num_layers=self.layer_num)

            self.value_down_prefix_hypernet = PrefixHyperNetLayerEmbedVer(embed_length=self.embed_length,
                                                                          embed_dim=self.hyper_embed_dim,
                                                                          mid_dim=self.hyper_mid_dim,
                                                                          target_mid_dim=self.mid_dim,
                                                                          output_dim=self.embed_dim,
                                                                          num_layers=self.layer_num)

            self.value_up_prefix_hypernet = PrefixHyperNetLayerEmbedVer(embed_length=self.embed_length,
                                                                        embed_dim=self.hyper_embed_dim,
                                                                        mid_dim=self.hyper_mid_dim,
                                                                        target_mid_dim=self.embed_dim,
                                                                        output_dim=self.mid_dim,
                                                                        num_layers=self.layer_num)

        else:
            self.key_down_prefix_hypernet = PrefixHyperNet(embed_length=self.embed_length,
                                                           embed_dim=self.hyper_embed_dim,
                                                           mid_dim=self.hyper_mid_dim, target_mid_dim=self.mid_dim,
                                                           output_dim=self.embed_dim,
                                                           num_layers=self.layer_num)

            self.key_up_prefix_hypernet = PrefixHyperNet(embed_length=self.embed_length, embed_dim=self.hyper_embed_dim,
                                                         mid_dim=self.hyper_mid_dim, target_mid_dim=self.embed_dim,
                                                         output_dim=self.mid_dim,
                                                         num_layers=self.layer_num)

            self.value_down_prefix_hypernet = PrefixHyperNet(embed_length=self.embed_length,
                                                             embed_dim=self.hyper_embed_dim,
                                                             mid_dim=self.hyper_mid_dim,
                                                             target_mid_dim=self.mid_dim,
                                                             output_dim=self.embed_dim,
                                                             num_layers=self.layer_num)

            self.value_up_prefix_hypernet = PrefixHyperNet(embed_length=self.embed_length,
                                                           embed_dim=self.hyper_embed_dim,
                                                           mid_dim=self.hyper_mid_dim, target_mid_dim=self.embed_dim,
                                                           output_dim=self.mid_dim,
                                                           num_layers=self.layer_num)

    def change_batch_size(self, flag, batch_size=None):
        if flag and batch_size is not None:
            self.batch_size = batch_size
        if flag:
            self.batch_size = int(self.batch_size / 2)

        else:
            self.batch_size = int(self.batch_size * 2)

    def forward(self, x):
        """
        x : [layer, batch, 197, 768]
        """
        key_prompts = []
        value_prompts = []

        prompt_shape = (
            self.batch_size, self.prompt_length, self.head_num, self.embed_dim)  # default (B, 10, 12, 64) * 12

        if self.prompt_init == "zero":
            key_prompt = nn.Parameter(torch.zeros(prompt_shape)).cuda()
            value_prompt = nn.Parameter(torch.zeros(prompt_shape)).cuda()

        elif self.prompt_init == "uniform":
            key_prompt = nn.Parameter(torch.randn(prompt_shape)).cuda()
            value_prompt = nn.Parameter(torch.randn(prompt_shape)).cuda()
            nn.init.uniform_(key_prompt, -1, 1)
            nn.init.uniform_(value_prompt, -1, 1)

        else:
            key_prompt = None
            value_prompt = None

        # calculate key
        key_weight_down_list, key_bias_down_list = self.key_down_prefix_hypernet(x,
                                                                                 self.batch_size)  # (12, B, 16, 64), (12, B, 16)
        key_weight_up_list, key_bias_up_list = self.key_up_prefix_hypernet(x,
                                                                           self.batch_size)  # (12, B, 64, 16), (12, B, 64)

        # calculate value
        value_weight_down_list, value_bias_down_list = self.value_down_prefix_hypernet(
            x, self.batch_size)  # (12, B, 16, 64), (12, B, 16)
        value_weight_up_list, value_bias_up_list = self.value_up_prefix_hypernet(x,
                                                                                 self.batch_size)  # (12, B, 64, 16), (12, B, 64)

        for i in range(self.layer_num):
            key_weight_down = key_weight_down_list[i]
            key_bias_down = key_bias_down_list[i]
            key_weight_up = key_weight_up_list[i]
            key_bias_up = key_bias_up_list[i]

            value_weight_down = value_weight_down_list[i]
            value_bias_down = value_bias_down_list[i]
            value_weight_up = value_weight_up_list[i]
            value_bias_up = value_bias_up_list[i]

            key_mid = apply_projection(key_prompt, key_weight_down, key_bias_down, self.batch_size)
            key_mid = F.relu(key_mid)
            key_result = apply_projection(key_mid, key_weight_up, key_bias_up, self.batch_size)

            # layer norm
            key_result = self.prefix_layer_norm(key_result)

            key_prompts.append(key_result)

            value_mid = apply_projection(value_prompt, value_weight_down, value_bias_down, self.batch_size)
            value_mid = F.relu(value_mid)
            value_result = apply_projection(value_mid, value_weight_up, value_bias_up, self.batch_size)

            # layer norm
            value_result = self.prefix_layer_norm(value_result)

            value_prompts.append(value_result)

        key_prompts = torch.stack(key_prompts).cuda()  # (m, l, h, dh) default (B, 12, 10, 12 ,64)
        value_prompts = torch.stack(value_prompts).cuda()  # (m, l, h, dh) default (B, 12, 10, 12 ,64)

        return key_prompts, value_prompts


def apply_projection(input_data, weights, biases, batch_size):
    _, height, width, channels = input_data.shape
    input_reshaped = input_data.view(batch_size, height * width, channels)

    weights = weights.permute(0, 2, 1)

    projected = torch.bmm(input_reshaped, weights)
    projected += biases.unsqueeze(1).expand_as(projected)

    output = projected.view(batch_size, height, width, -1)

    return output


def init_linear_layer(linear_layer, std=1e-2):
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)


def linear_layer(input_dim, output_dim, std=1e-2):
    linear = nn.Linear(input_dim, output_dim)
    init_linear_layer(linear, std=std)
    return linear


class PrefixHyperNet(nn.Module):
    def __init__(self, embed_length, embed_dim, mid_dim, target_mid_dim, output_dim, num_layers):
        super(PrefixHyperNet, self).__init__()

        self.embed_length = embed_length
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim
        self.target_mid_dim = target_mid_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        self.weight_generator = nn.Sequential(
            linear_layer(self.embed_dim, self.mid_dim),
            nn.ReLU(),
            linear_layer(self.mid_dim, self.target_mid_dim * self.output_dim))
        self.bias_generator = nn.Sequential(
            linear_layer(self.embed_dim, self.mid_dim),
            nn.ReLU(),
            linear_layer(self.mid_dim, self.target_mid_dim))

    def forward(self, embeddings, batch_size):
        weights = []
        biases = []

        for layer_idx in range(self.num_layers):
            input_embedding = embeddings[layer_idx]
            input_embedding = self.layer_norm(input_embedding)
            input_embedding = torch.mean(input_embedding, dim=1)

            weight = self.weight_generator(input_embedding).view(-1, self.target_mid_dim, self.output_dim).cuda()
            bias = self.bias_generator(input_embedding).view(batch_size, -1).cuda()

            weights.append(weight)
            biases.append(bias)

        return weights, biases


class PrefixHyperNetLayerEmbedVer(nn.Module):
    def __init__(self, embed_length, embed_dim, mid_dim, target_mid_dim, output_dim, num_layers):
        super(PrefixHyperNetLayerEmbedVer, self).__init__()

        self.embed_length = embed_length
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim
        self.target_mid_dim = target_mid_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.layer_embedding = nn.Embedding(num_layers, embed_dim)

        self.weight_generator = nn.Sequential(
            linear_layer(self.embed_dim * 2, self.mid_dim),
            nn.ReLU(),
            linear_layer(self.mid_dim, self.target_mid_dim * self.output_dim))
        self.bias_generator = nn.Sequential(
            linear_layer(self.embed_dim * 2, self.mid_dim),
            nn.ReLU(),
            linear_layer(self.mid_dim, self.target_mid_dim))

    def forward(self, embeddings, batch_size):
        weights = []
        biases = []

        for layer_idx in range(self.num_layers):
            input_embedding = embeddings[layer_idx]
            input_embedding = torch.mean(input_embedding, dim=1)

            layer_embed = self.layer_embedding(torch.tensor([layer_idx], device=input_embedding.device)).expand_as(
                input_embedding)

            combined_embedding = torch.cat((input_embedding, layer_embed), dim=1)

            weight = self.weight_generator(combined_embedding).view(-1, self.target_mid_dim, self.output_dim).cuda()
            bias = self.bias_generator(combined_embedding).view(batch_size, -1).cuda()

            weights.append(weight)
            biases.append(bias)

        return weights, biases
