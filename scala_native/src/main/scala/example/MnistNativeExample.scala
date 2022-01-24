package example

import me.shadaj.scalapy.interpreter.{CPythonInterpreter, Platform, PyValue}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{Any, ConvertableToSeqElem, PyQuote, SeqConverters}
import me.shadaj.scalapy.readwrite.Writer

object MnistExample extends Greeting with App {
  val torch = py.module("torch")
  val nn = py.module("torch.nn")
  val F = py.module("torch.nn.functional")
  val T = py.module("torchvision.transforms")
  val optim = py.module("torch.optim")
  val orderedDict = py.module("collections", "OrderedDict")
  val datasets = py.module("torchvision.datasets")
  val resnet50 = py.module("torchvision.models", "resnet50")
  val train_test_split = py.module("sklearn.model_selection", "train_test_split")

  val transforms = T.Compose(List(T.ToTensor(), T.Normalize(
    List(0.1307).toPythonProxy,
    List(0.3081).toPythonProxy)).toPythonProxy)

  val dataset_train = datasets.MNIST("./data", train = true, download = true, transform = transforms)
  val dataset_test = datasets.MNIST("./data", train = false, transform = transforms)

  val train_loader = torch.utils.data.DataLoader(dataset_train, 64)
  val test_loader = torch.utils.data.DataLoader(dataset_test, 1000)

  // TODO:fixme
  CPythonInterpreter.execManyLines(
    """
      |import torch
      |from torch import nn
      |import torch.nn.functional as F
      |
      |callback=None
      |def callback_updater(call):
      |    global callback
      |    callback = call
      |
      |
      |class Net(nn.Module):
      |    def __init__(self):
      |        super(Net, self).__init__()
      |        self.conv1 = nn.Conv2d(1, 32, 3, 1)
      |        self.conv2 = nn.Conv2d(32, 64, 3, 1)
      |        self.dropout1 = nn.Dropout(0.25)
      |        self.dropout2 = nn.Dropout(0.5)
      |        self.fc1 = nn.Linear(9216, 128)
      |        self.fc2 = nn.Linear(128, 10)
      |
      |        self.forward_calls=0
      |
      |
      |    def forward(self, x):
      |        x = self.conv1(x)
      |        x = F.relu(x)
      |        x = self.conv2(x)
      |        x = F.relu(x)
      |        x = F.max_pool2d(x, 2)
      |        x = self.dropout1(x)
      |        x = torch.flatten(x, 1)
      |        x = self.fc1(x)
      |        x = F.relu(x)
      |        x = self.dropout2(x)
      |        x = self.fc2(x)
      |        output = F.log_softmax(x, dim=1)
      |
      |        self.forward_calls=self.forward_calls+1
      |        if self.forward_calls % 100==0 and callback:
      |            callback(self.forward_calls)
      |
      |        return output
      |        """.stripMargin
  )
  val model = py"Net()"
  val optimizer = optim.Adadelta(model.parameters(), lr = 1.0)
  val lr_scheduler = py.module("torch.optim.lr_scheduler")
  val scheduler = lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.7)
  val device = torch.device("cpu") // or "GPU"

  def reactOnCallback(forwardCallsCounter: Int) = {
    println(forwardCallsCounter)
  }
  //TODO: fixme missing updateDynamic for global scope this will not work =>
  //py.Dynamic.global.callback = reactOnCallback
  // using hack
  //py.Dynamic.global.callback_updater(reactOnCallback)

  def train(model: py.Dynamic, device: py.Dynamic, train_loader: py.Dynamic, optimizer: py.Dynamic, epoch: Int, log_interval: Int = 100) = {
    model.train()

    val trainLen = len(train_loader.dataset)
    for {
      tupleData <- toTraversable(train_loader.__iter__()).toSeq.zipWithIndex
      (data, batch_idx) = tupleData
    } {

      val input_data = data.bracketAccess(0).to(device)
      val target = data.bracketAccess(1).to(device)
      optimizer.zero_grad()
      val output = model(input_data)
      val loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      optimizer.step()

      if (batch_idx % log_interval == 0) {
        val percentDone = (100.0 * batch_idx * len(input_data) / trainLen)
        println(
          f"Train Epoch: ${epoch} [${batch_idx * len(input_data)}/$trainLen ($percentDone%.1f%%)] Loss: ${loss.item().as[Double]}%.6f")
      }
    }
  }

  def test(model: py.Dynamic, device: py.Dynamic, test_loader: py.Dynamic) = {
    model.eval()
    py.`with`(torch.no_grad()) { _ =>
      var testLoss: Double = 0
      var correct: Int = 0

      val testLen = len(test_loader.dataset)

      for (testData <- toTraversable(test_loader.__iter__())) {
        val data = testData.bracketAccess(0).to(device)
        val target = testData.bracketAccess(1).to(device)
        val output = model(data)

        testLoss += F.nll_loss(output, target, reduction = "sum").item().as[Double]

        val pred = output.argmax(dim = 1, keepdim = true)
        val view = target.view_as(pred)
        val curr_correct = pred.applyDynamicNamed("eq")(("", view)).sum().item().as[Int] // TODO: fixme: conflict of namespaces
        correct += curr_correct
      }
      testLoss /= testLen
      println(s"Test set: Average loss: ${testLoss}, Accuracy: ${correct}/${testLen} (${100.0 * correct / testLen % .2f})")
    }
  }

  for (epoch <- (1 to 1)) {
    timeIt {
      train(model = model, device = device, train_loader = train_loader, optimizer = optimizer, epoch = epoch)
      test(model, device, test_loader)
      scheduler.step()
    }
  }
}


trait Greeting {
  lazy val greeting: String = "hello"
}

