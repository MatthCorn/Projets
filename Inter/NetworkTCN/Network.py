import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from Complete.Transformer.LearnableModule import LearnableParameters


class Chomp1d(nn.Module):
    """Coupe le padding à droite pour garantir la causalité stricte."""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Un bloc résiduel TCN standard.
    Il contient 2 convolutions pour permettre la non-linéarité et la connexion résiduelle.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # Conv 1
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        # Conv 2
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # Adaptation de dimension pour le Residual (si input != output)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.GELU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Le réseau complet. Empile des TemporalBlocks selon la liste num_channels.
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)

        # Calcul du Champ Récepteur total (Receptive Field)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1)
        print(f"TCN Configuré: {num_levels} blocs. Receptive Field: {self.receptive_field} steps.")

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # Le padding dépend de la dilation pour garder la taille temporelle identique
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MemoryUpdateTCN(nn.Module):
    def __init__(
            self,
            input_dim_1,
            input_dim_2,
            hidden_dim,  # Taille interne des embeddings
            output_dim,
            # C'est ici que vous réglez la puissance du TCN :
            tcn_channels=[64, 64, 64, 64, 64],  # 5 blocs => dilation jusqu'à 16
            kernel_size=3,
            dropout=0.0,
            use_layernorm=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_layernorm = use_layernorm

        # --- 1. EMBEDDINGS (Votre logique spécifique) ---
        self.embedding_1 = nn.Sequential(
            nn.Linear(input_dim_1, hidden_dim), nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.embedding_2 = nn.Sequential(
            nn.Linear(input_dim_2, hidden_dim), nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.merge_embeddings = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

        self.next_token = LearnableParameters(torch.normal(0, 1, [1, 1, hidden_dim]))

        # --- 2. LE CŒUR TCN (Remplace LSTM + Attention) ---
        # Note: input du TCN = hidden_dim (sortie du merge)
        self.tcn = TCN(num_inputs=hidden_dim,
                       num_channels=tcn_channels,
                       kernel_size=kernel_size,
                       dropout=dropout)

        # On récupère le champ récepteur pour gérer le buffer
        self.max_history = self.tcn.receptive_field

        if use_layernorm:
            # On normalise la sortie du TCN (c'est souvent utile)
            # Attention: LayerNorm sur (B, C, T) ou (B, T, C) -> ici on le fera sur la dernière dim
            self.post_tcn_ln = nn.LayerNorm(tcn_channels[-1])

        # --- 3. SORTIE ---
        # On projette la sortie du TCN (qui a pour dim le dernier canal de tcn_channels)
        self.head = nn.Linear(tcn_channels[-1], output_dim)

    def forward(self, input_1, input_2, next_mask=None):
        """Mode Entraînement (Parallèle)"""
        if next_mask is None:
            next_mask = torch.zeros(*input_1.shape[:-1], 1).to(input_1.device)

        # A. Préparation des entrées
        x_1 = self.embedding_1(input_1)
        x_2 = self.embedding_2(input_2)
        x_2 = x_2 * (1 - next_mask) + next_mask * self.next_token()

        # Fusion -> [Batch, Time, Hidden]
        x = self.merge_embeddings(torch.cat([x_1, x_2], dim=-1))

        # B. Passage TCN
        # Le TCN (Conv1d) veut [Batch, Channels, Time]
        x_perm = x.transpose(1, 2)

        tcn_out = self.tcn(x_perm)

        # Retour en [Batch, Time, Channels]
        tcn_out = tcn_out.transpose(1, 2)

        if self.use_layernorm:
            tcn_out = self.post_tcn_ln(tcn_out)

        # C. Sortie directe (Pas d'attention, pas de combinaison complexe)
        pred = self.head(tcn_out)

        # Calcul optionnel de distance (votre logique existante)
        next_dist = torch.norm(tcn_out - self.next_token(), dim=-1, keepdim=True) / torch.norm(self.next_token())

        return pred, next_dist

    def step(self, input_1, input_2, buffer=None, next_mask=None):
        """
        Mode Inférence Pas-à-Pas (Bufferisé).
        buffer: contient l'historique brut des embeddings [Batch, Hidden, T_history]
        """
        # 1. Embeddings de l'instant t
        if next_mask is None:
            next_mask = torch.zeros(*input_1.shape[:-1], 1, 1).to(input_1.device)

        x_1 = self.embedding_1(input_1.unsqueeze(1))
        x_2 = self.embedding_2(input_2.unsqueeze(1))
        x_2 = x_2 * (1 - next_mask) + next_mask * self.next_token()

        # x_t : [Batch, 1, Hidden]
        x_t = self.merge_embeddings(torch.cat([x_1, x_2], dim=-1))

        # On permute pour le format buffer TCN: [Batch, Hidden, 1]
        x_t_perm = x_t.transpose(1, 2)

        # 2. Gestion du Buffer
        if buffer is None:
            buffer = x_t_perm
        else:
            buffer = torch.cat([buffer, x_t_perm], dim=2)

        # On coupe le buffer pour ne pas exploser la mémoire inutilement
        # On garde juste assez pour calculer la convolution la plus large
        if buffer.size(2) > self.max_history:
            buffer = buffer[:, :, -self.max_history:]

        # 3. Inférence TCN
        # On applique le TCN sur tout le buffer
        tcn_out_seq = self.tcn(buffer)  # [Batch, Channels, Buffer_Len]

        # On ne prend que le DERNIER point temporel (l'instant présent)
        h_t = tcn_out_seq[:, :, -1]  # [Batch, Channels]

        if self.use_layernorm:
            h_t = self.post_tcn_ln(h_t)

        # 4. Sortie
        y_t = self.head(h_t)  # [Batch, Output_Dim]

        # (Pour compatibilité avec votre code existant qui attend 4 sorties)
        # Ici next_dist est calculé sur le vecteur courant
        next_dist = torch.norm(h_t - self.next_token().squeeze(0).squeeze(0), dim=-1) / torch.norm(self.next_token())

        # On retourne le buffer mis à jour au lieu de (hidden, H_past)
        # H_past n'est plus utile car pas d'attention
        return y_t, buffer, None, next_dist

if __name__ == '__main__':
    N = MemoryUpdateTCN(10,
            10,
            64,  # Taille interne des embeddings
            5,
            # C'est ici que vous réglez la puissance du TCN :
            tcn_channels=[64, 64, 64, 64, 64],  # 5 blocs => dilation jusqu'à 16
            kernel_size=3,
            dropout=0.0,
            use_layernorm=True,
    )
    x1 = torch.normal(0, 1, (40, 20, 10))
    x2 = torch.normal(0, 1, (40, 20, 10))
    y1, next_dist_1 = N(x1, x2)

    y_list = []
    next_dist_list = []
    hidden = None
    H_past = None
    for k in range(x1.shape[1]):
        x1_k = x1[:, k]
        x2_k = x2[:, k]
        y_k, hidden, H_past, next_dist = N.step(x1_k, x2_k, hidden, H_past)
        y_list.append(y_k.unsqueeze(1))
        next_dist_list.append(next_dist.unsqueeze(1))
    y2 = torch.cat(y_list, dim=1)
    next_dist_2 = torch.cat(next_dist_list, dim=1)

    print(y2[0, :, 0])
    print(y1[0, :, 0])

    print(next_dist_1[0, :, 0])
    print(next_dist_2[0])