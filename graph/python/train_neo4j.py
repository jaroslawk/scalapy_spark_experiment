import pandas as pd
import torch
import torch.nn.functional as F
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.transforms import ToUndirected, RandomLinkSplit


def fetch_data(query, params={}) -> pd.DataFrame:
    with driver.session() as session:
        result = session.run(query, params)
        return pd.DataFrame([r.values() for r in result], columns=result.keys())


def load_node(cypher, index_col, encoders=None, **kwargs):
    # Execute the cypher query and retrieve data from Neo4j
    df = fetch_data(cypher)
    df.set_index(index_col, inplace=True)
    # Define node mapping
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    # Define node features
    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge(cypher, src_index_col, src_mapping, dst_index_col, dst_mapping, encoders=None, **kwargs):
    # Execute the cypher query and retrieve data from Neo4j
    df = fetch_data(cypher)
    # Define edge index
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]

    edge_index = torch.tensor([src, dst])
    # Define edge features
    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


class SequenceEncoder(object):
    # The 'SequenceEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()


class GenresEncoder(object):
    # The 'GenreEncoder' splits the raw column strings by 'sep' and converts
    # individual elements to categorical labels.
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x


class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None, is_list=False):
        self.dtype = dtype
        self.is_list = is_list

    def __call__(self, df):
        if self.is_list:
            return torch.stack([torch.tensor(el) for el in df.values])
        return torch.from_numpy(df.values).to(self.dtype)


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


def train():
    model.train()
    optimizer.zero_grad()
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data['user', 'rates', 'movie'].edge_label_index)
    target = train_data['user', 'rates', 'movie'].edge_label
    loss = weighted_mse_loss(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['user', 'rates', 'movie'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data['user', 'rates', 'movie'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)


if __name__ == "__main__":
    url = 'bolt://18.213.193.202:7687'
    user = 'neo4j'
    password = 'shoulders-typewriters-men'

    driver = GraphDatabase.driver(url, auth=(user, password))

    '''data = fetch_data("""
    CALL gds.graph.create('movies', ['Movie', 'Person'], {ACTED_IN: {orientation:'UNDIRECTED'}, DIRECTED: {orientation:'UNDIRECTED'}})
    """)'''

    user_query = """
    MATCH (u:User) RETURN u.userId AS userId
    """

    user_x, user_mapping = load_node(user_query, index_col='userId')

    movie_query = """
    MATCH (m:Movie)-[:IN_GENRE]->(genre:Genre)
    WITH m, collect(genre.name) AS genres_list
    RETURN m.movieId AS movieId, m.title AS title, apoc.text.join(genres_list, '|') AS genres, m.fastrp AS fastrp
    """

    movie_x, movie_mapping = load_node(
        movie_query,
        index_col='movieId', encoders={
            'title': SequenceEncoder(),
            'genres': GenresEncoder(),
            'fastrp': IdentityEncoder(is_list=True)
        })

    rating_query = """
    MATCH (u:User)-[r:RATED]->(m:Movie) 
    RETURN u.userId AS userId, m.movieId AS movieId, r.rating AS rating
    """

    edge_index, edge_label = load_edge(
        rating_query,
        src_index_col='userId',
        src_mapping=user_mapping,
        dst_index_col='movieId',
        dst_mapping=movie_mapping,
        encoders={'rating': IdentityEncoder(dtype=torch.long)},
    )

    #########################
    # torch part
    #########################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = HeteroData()
    # Add user node features for message passing:
    data['user'].x = torch.eye(len(user_mapping), device=device)
    # Add movie node features
    data['movie'].x = movie_x
    # Add ratings between users and movies
    data['user', 'rates', 'movie'].edge_index = edge_index
    data['user', 'rates', 'movie'].edge_label = edge_label
    data.to(device, non_blocking=True)

    data = ToUndirected()(data)
    del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.

    # 2. Perform a link-level split into training, validation, and test edges.
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=0.0,
        edge_types=[('user', 'rates', 'movie')],
        rev_edge_types=[('movie', 'rev_rates', 'user')],
    )
    train_data, val_data, test_data = transform(data)

    weight = torch.bincount(train_data['user', 'movie'].edge_label)
    weight = weight.max() / weight

    model = Model(hidden_channels=64).to(device)

    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    with torch.no_grad():
        model.encoder(train_data.x_dict, train_data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 300):
        loss = train()
        train_rmse = test(train_data)
        val_rmse = test(val_data)
        test_rmse = test(test_data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
              f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')

    num_movies = len(movie_mapping)
    num_users = len(user_mapping)

    reverse_movie_mapping = dict(zip(movie_mapping.values(), movie_mapping.keys()))
    reverse_user_mapping = dict(zip(user_mapping.values(), user_mapping.keys()))

    results = []

    for user_id in range(0, num_users):
        row = torch.tensor([user_id] * num_movies)
        col = torch.arange(num_movies)
        edge_label_index = torch.stack([row, col], dim=0)

        pred = model(data.x_dict, data.edge_index_dict,
                     edge_label_index)
        pred = pred.clamp(min=0, max=5)

        user_neo4j_id = reverse_user_mapping[user_id]

        mask = (pred == 5).nonzero(as_tuple=True)

        ten_predictions = [reverse_movie_mapping[el] for el in mask[0].tolist()[:10]]
        results.append({'user': user_neo4j_id, 'movies': ten_predictions})
