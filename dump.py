class _DualGNN(nn.Module):
    """
    GNN model that processes chromophore and solvent separately,
    then combines their representations to predict absorption and emission.
    Supports multiple GNN architectures: GCN, GAT, GIN, SchNet
    """

    def __init__(self, node_features=7, hidden_dim=64, output_dim=2,
                 use_solvent=True, gnn_type='gcn', num_layers=2):
        super(DualGNN, self).__init__()

        self.use_solvent = use_solvent
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # ---- SchNet path (treat as full model, not a conv layer stack) ----
        if self.gnn_type == 'schnet':
            # SchNet returns a graph-level prediction of shape [batch_size, 1]
            # (per PyG implementation; it's a regression head internally).
            self.schnet_chromo = SchNet(
                hidden_channels=hidden_dim,
                num_filters=hidden_dim,
                num_interactions=num_layers,
                # keep defaults for num_gaussians/cutoff unless you want to tune them
                readout='add'
            )
            self.schnet_solvent = None
            if self.use_solvent:
                self.schnet_solvent = SchNet(
                    hidden_channels=hidden_dim,
                    num_filters=hidden_dim,
                    num_interactions=num_layers,
                    readout='add'
                )

            # Combine chromo_pred (1) + solvent_pred (1) -> 2 features -> output_dim (2)
            in_dim = 2 if self.use_solvent else 1
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(0.2)
            return  # important: don't build the other conv stacks

        # ---- Non-SchNet path (GCN/GAT/GIN) ----
        self.chromo_convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = node_features if i == 0 else hidden_dim
            if gnn_type == 'gcn' or gnn_type == 'gcn+super_node':
                self.chromo_convs.append(GCNConv(in_dim, hidden_dim))
            elif gnn_type == 'gat':
                heads = 4
                out_dim = hidden_dim // heads
                self.chromo_convs.append(GATConv(in_dim, out_dim, heads=heads, concat=True))
            elif gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.chromo_convs.append(GINConv(mlp))

        if use_solvent:
            self.solvent_convs = nn.ModuleList()
            for i in range(num_layers):
                in_dim = node_features if i == 0 else hidden_dim
                if gnn_type == 'gcn' or gnn_type == 'gcn+super_node':
                    self.solvent_convs.append(GCNConv(in_dim, hidden_dim))
                elif gnn_type == 'gat':
                    heads = 4
                    out_dim = hidden_dim // heads
                    self.solvent_convs.append(GATConv(in_dim, out_dim, heads=heads, concat=True))
                elif gnn_type == 'gin':
                    mlp = nn.Sequential(
                        nn.Linear(in_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                    self.solvent_convs.append(GINConv(mlp))

            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        if use_solvent:
            self.solvent_batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

    def forward(self, chromo_data, solvent_data):
        # ---- SchNet forward ----
        if self.gnn_type == 'schnet':
            # SchNet input: z (atomic numbers), pos, batch
            z_c = chromo_data.x[:, 0].long()
            y_c = self.schnet_chromo(z_c, chromo_data.pos, chromo_data.batch)  # [B, 1]

            if self.use_solvent:
                z_s = solvent_data.x[:, 0].long()
                y_s = self.schnet_solvent(z_s, solvent_data.pos, solvent_data.batch)  # [B, 1]
                x = torch.cat([y_c, y_s], dim=1)  # [B, 2]
            else:
                x = y_c  # [B, 1]

            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)  # [B, 2]
            return x

        # ---- Non-SchNet forward ----
        x_c, edge_index_c, batch_c = chromo_data.x, chromo_data.edge_index, chromo_data.batch
        for i, conv in enumerate(self.chromo_convs):
            x_c = conv(x_c, edge_index_c)
            x_c = self.batch_norms[i](x_c)
            x_c = F.relu(x_c)
            x_c = self.dropout(x_c)

        if self.gnn_type == 'gcn+super_node':
            indexes = torch.cat([
                torch.where(batch_c[1:] != batch_c[:-1])[0],
                batch_c.new_tensor([batch_c.numel() - 1])
            ])
            x_c = x_c[indexes]
        elif self.gnn_type == 'gin':
            x_c = global_add_pool(x_c, batch_c)
        else:
            x_c = global_mean_pool(x_c, batch_c)

        if self.use_solvent:
            x_s, edge_index_s, batch_s = solvent_data.x, solvent_data.edge_index, solvent_data.batch
            for i, conv in enumerate(self.solvent_convs):
                x_s = conv(x_s, edge_index_s)
                x_s = self.solvent_batch_norms[i](x_s)
                x_s = F.relu(x_s)
                x_s = self.dropout(x_s)

            if self.gnn_type == 'gcn+super_node':
                indexes = torch.cat([
                    torch.where(batch_s[1:] != batch_s[:-1])[0],
                    batch_s.new_tensor([batch_s.numel() - 1])
                ])
                x_s = x_s[indexes]
            elif self.gnn_type == 'gin':
                x_s = global_add_pool(x_s, batch_s)
            else:
                x_s = global_mean_pool(x_s, batch_s)

            x = torch.cat([x_c, x_s], dim=1)
        else:
            x = x_c

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
