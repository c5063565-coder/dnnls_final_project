class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size = emb_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        emb = self.embedding(x)
        out, (h, c) = self.lstm(emb)
        return out, h, c
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe= torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
class EncoderTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim,
                 num_layers=2, nhead=4, dropout=0.1, max_len=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.input_proj = (
            nn.Linear(emb_dim, hidden_dim)
            if emb_dim != hidden_dim
            else nn.Identity()
        )
        self.pos_enc = PositionalEncoding(hidden_dim, max_len, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model= hidden_dim,
            nhead = nhead,
            dim_feedforward = hidden_dim * 4,
            dropout = dropout,
            batch_first = True,
            norm_first = True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x):
        pad_mask = (x == 0)
        emb = self.embedding(x)
        proj = self.input_proj(emb)
        x = self.pos_enc(proj)
        out = self.transformer(
            x,
            src_key_padding_mask = pad_mask
        )
        out = self.out_proj(out)
        mask_f = (~pad_mask).unsqueeze(-1).float()
        pooled = (out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        h = pooled.unsqueeze(0)
        c = torch.zeros_like(h)
        return out, h, c
    def get_attention_weights(self, x, max_tokens=20):
        x = x[:, :max_tokens]
        pad_mask = (x == 0)
        emb = self.embedding(x)
        proj = self.input_proj(emb)
        enc = self.pos_enc(proj)
        layer = self.transformer.layers[0]
        with torch.no_grad():
            _, attn_weights = layer.self_attn(
                enc, enc, enc,
                key_padding_mask = pad_mask,
                need_weights = True,
                average_attn_weights = False
            )
        return attn_weights
class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size = emb_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0
        )
        self.out = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, h, c):
        emb = self.embedding(x)
        out, (h, c) = self.lstm(emb, (h, c))
        predictions = self.out(out)
        return predictions, h, c
class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        _, h, c = self.encoder(src)
        preds, _, _ = self.decoder(tgt[:, :-1], h, c)
        return preds
class Backbone(nn.Module):
    def __init__(self, latent_dim=16, output_w=8, output_h=16):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=2, padding=3),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1),

            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1),
        )
        self.flatten_dim = 64 * output_w * output_h
        self.fc1 = nn.Sequential(
            nn.Linear(self.flatten_dim, latent_dim),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.encoder_conv(x)
        x = x.view(-1, self.flatten_dim)
        return self.fc1(x)
class ResNetBackbone(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
    def forward(self, x):
        features = self.features(x)
        return self.projection(features)
class VisualEncoder(nn.Module):
    def __init__(self, latent_dim=16, output_w=8, output_h=16):
        super().__init__()
        if USE_RESNET_ENCODER:
            self.content_backbone = ResNetBackbone(latent_dim)
            self.context_backbone = ResNetBackbone(latent_dim)
            print("  VisualEncoder: using ResNet-18 backbone")
        else:
            self.content_backbone = Backbone(latent_dim, output_w, output_h)
            self.context_backbone = Backbone(latent_dim, output_w, output_h)
            print("  VisualEncoder: using CNN baseline backbone")

        self.projection = nn.Linear(2 * latent_dim, latent_dim)
    def forward(self, x):
        z_content = self.content_backbone(x)
        z_context = self.context_backbone(x)
        z = torch.cat([z_content, z_context], dim=1)
        return self.projection(z)
class VisualDecoder(nn.Module):
    def __init__(self, latent_dim=16, output_w=8, output_h=16):
        super().__init__()
        self.flatten_dim = 64 * output_w * output_h
        self.output_w = output_w
        self.output_h = output_h
        self.imh = 60
        self.imw = 125
        self.fc1 = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=(1,1)),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2, output_padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 3, 7, stride=2, padding=3, output_padding=(1,1)),
            nn.Sigmoid()
        )
    def forward(self, z):
        x = self.fc1(z)
        x_content = self.decode_image(x)
        x_context = self.decode_image(x)
        return x_content, x_context
    def decode_image(self, x):
        x = x.view(-1, 64, self.output_w, self.output_h)
        x = self.decoder_conv(x)
        return x[:, :, :self.imh, :self.imw]
class VisualAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = VisualEncoder(latent_dim)
        self.decoder = VisualDecoder(latent_dim)
    def forward(self, x):
        z = self.encoder(x)
        x_c, x_ctx = self.decoder(z)
        return x_c, x_ctx
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn    = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, rnn_out):
        scores  = self.attn(rnn_out)
        scores  = scores.squeeze(2)
        weights = self.softmax(scores)
        context = torch.bmm(
            weights.unsqueeze(1),
            rnn_out
        ).squeeze(1)
        return context
class SequencePredictor(nn.Module):
    def __init__(self, visual_ae, text_ae, latent_dim, gru_hidden):
        super().__init__()
        self.image_encoder = visual_ae.encoder
        self.text_encoder  = text_ae.encoder
        fusion_dim = latent_dim * 2
        self.temporal_rnn  = nn.GRU(
            input_size  = fusion_dim,
            hidden_size = latent_dim,
            batch_first = True
        )
        self.attention = Attention(gru_hidden)
        self.projection = nn.Sequential(
            nn.Linear(gru_hidden * 2, latent_dim),
            nn.ReLU()
        )
        self.image_decoder = visual_ae.decoder
        self.text_decoder  = text_ae.decoder
        self.fused_to_h0 = nn.Linear(latent_dim, latent_dim)
        self.fused_to_c0 = nn.Linear(latent_dim, latent_dim)

    def forward(self, image_seq, text_seq, target_seq):
        B, S, C, H, W = image_seq.shape
        img_flat = image_seq.view(B*S, C, H, W)
        txt_flat = text_seq.view(B*S, -1)
        z_v_flat = self.image_encoder(img_flat)
        _, h_t, c_t = self.text_encoder(txt_flat)
        z_t_flat    = h_t.squeeze(0)
        z_v_seq = z_v_flat.view(B, S, -1)
        z_t_seq = z_t_flat.view(B, S, -1)
        fusion = torch.cat([z_v_seq, z_t_seq], dim=2)
        gru_out, _ = self.temporal_rnn(fusion)
        context = self.attention(gru_out)
        h_last = gru_out[:, -1, :]
        fused = self.projection(
            torch.cat([h_last, context], dim=1)
        )
        pred_img_content, pred_img_context = self.image_decoder(fused)
        h0 = self.fused_to_h0(fused).unsqueeze(0)
        c0 = self.fused_to_c0(fused).unsqueeze(0)
        pred_text, _, _ = self.text_decoder(
            target_seq.squeeze(1)[:, :-1], h0, c0
        )
        return (
            pred_img_content,
            pred_img_context,
            pred_text,
            h0,
            c0,
            z_v_seq,
            z_t_seq
        )
