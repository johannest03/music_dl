import datetime
from Pathlib import Path
import jax
from torch.utils.tensorboard import SummaryWriter
from orbax.checkpoint import Checkpointer, ModelCheckpoint


class Trainer():

    def __init__(self, model, dataloader, validation_dataloader, optimizer, loss_fn, log_dir, ckpt_dir):
        self.model = model
        self.dataloader = dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        log_dir = Path(log_dir).resolve() / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = Path(ckpt_dir).resolve() / timestamp
        ckpt_path.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir)

        self.checkpointer = ModelCheckpoint(ckpt_path)


    def compile(self, rng, input_shape):
        params = self.model.init(rng, jax.numpy.ones(input_shape))
        self.opt_state = self.optimizer.init(params)
        self.params = params

    @jax.jit
    def train_step(self, batch):
        inputs, _ = batch  # batch = (input_seq, name_of_song), targets created below
        targets = inputs[:, 1:]
        input_seq = inputs[:, :-1]
    
        def loss_fn(params):
            logits = self.model.apply(params, input_seq)
            # Assume logits shape: (batch, seq_len, vocab_size), targets: (batch, seq_len)
            return self.loss_fn(logits, targets)

        grads = jax.grad(loss_fn)(self.params)
        loss = loss_fn(self.params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        self.params = jax.tree_util.tree_multimap(lambda p, u: p + u, self.params, updates)
        return loss

    @jax.jit
    def eval_step(self, batch):
        inputs, _ = batch
        targets = inputs[:, 1:]
        input_seq = inputs[:, :-1]

        def loss_fn(params):
            logits = self.model.apply(params, input_seq)
            return self.loss_fn(logits, targets)

        accuracy = jax.numpy.mean(jax.numpy.argmax(self.model.apply(self.params, input_seq), axis=-1) == targets)

        return loss_fn(self.params), accuracy

    def fit(self, epochs, batch_size=None):
        assert self.params is not None, "Model is not compiled. Please call compile() before fit()."

        if batch_size is not None:
            self.dataloader.set_batch_size(batch_size)

        for epoch in range(epochs):
            for batch, song_names in self.dataloader:
                loss = self.train_step(batch)

            self.writer.add_scalar('Loss/train', loss, self.global_step)

            if self.validation_dataloader and epoch % 4 == 0:
                val_loss = 0
                val_accuracy = 0
                for val_batch, val_song_names in self.validation_dataloader:
                    batch_loss, batch_accuracy = self.eval_step(val_batch)
                    val_accuracy += batch_accuracy
                    val_loss += batch_loss
                val_loss /= len(self.validation_dataloader)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)

            # Save checkpoints periodically
            if epoch % 10 == 0:
                self.checkpointer.save(self.params, step=epoch)
