package example

import me.shadaj.scalapy.interpreter.{CPythonInterpreter, Platform, PyValue}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{Any, ConvertableToSeqElem, PyQuote, SeqConverters}
import me.shadaj.scalapy.readwrite.Writer

object GraphExample extends Greeting with App {
  val torch = py.module("torch")
  val ge = py.module("torch_geometric.datasets")
  val T = py.module("torch_geometric.transforms")

  val device = torch.device(if (torch.cuda.is_available().as[Boolean]) "gpu" else "cpu")
  execute("model.py")
  val dataset = ge.MovieLens("./data/MovieLens", model_name="all-MiniLM-L6-v2")
  val data = dataset.bracketAccess(0).to(device)

  // Add user node features for message passing:
  data.bracketAccess("user").x = torch.eye(data.bracketAccess("user").num_nodes, device=device)
  del(data.bracketAccess("user").num_nodes)

  // Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
  val data_uni = T.ToUndirected()(data)
  del(data_uni.bracketAccess("movie", "rev_rates", "user").edge_label)  // Remove "reverse" label.

  // Perform a link-level split into training, validation, and test edges:
  val train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=List("user", "rates", "movie").toPythonProxy,
    rev_edge_types=List("movie", "rev_rates", "user").toPythonProxy,
  )(data)

  /*
    println(train_data)

  # We have an unbalanced dataset with many labels for rating 3 and 4, and very
  # few for 0 and 1. Therefore we use a weighted MSE loss.
  if args.use_weighted_loss:
    weight = torch.bincount(train_data['user', 'movie'].edge_label)
  weight = weight.max() / weight
  else:
  weight = None

  model = Model(hidden_channels=32).to(device)

  # Due to lazy initialization, we need to run one model step so the number
  # of parameters can be inferred:
  with torch.no_grad():
    model.encoder(train_data.x_dict, train_data.edge_index_dict)

  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


  def train():
  model.train()
  optimizer.zero_grad()
  pred = model(train_data.x_dict, train_data.edge_index_dict,
    train_data['user', 'movie'].edge_label_index)
  target = train_data['user', 'movie'].edge_label
  loss = weighted_mse_loss(pred, target, weight)
  loss.backward()
  optimizer.step()
  return float(loss)


  @torch.no_grad()
  def test(data):
  model.eval()
  pred = model(data.x_dict, data.edge_index_dict,
    data['user', 'movie'].edge_label_index)
  pred = pred.clamp(min=0, max=5)
  target = data['user', 'movie'].edge_label.float()
  rmse = F.mse_loss(pred, target).sqrt()
  return float(rmse)


  for epoch in range(1, 2): # 301
  loss = train()
  train_rmse = test(train_data)
  val_rmse = test(val_data)
  test_rmse = test(test_data)
  print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
  f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
  */
}


trait Greeting {
  lazy val greeting: String = "hello"
}

