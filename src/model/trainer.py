from datetime import datetime
from pathlib import Path
import jax
from torch.utils.tensorboard import SummaryWriter
from music_utils.midi.midi_decoder import MidiDecoder
from orbax.checkpoint import StandardCheckpointHandler
import optax
from params import OUTPUT_PATH
from tqdm import tqdm

class Trainer():

    def __init__(self, model, dataloader, validation_dataloader, optimizer, loss_fn, log_dir, ckpt_dir, sample_path):
        self.model = model
        self.dataloader = dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optimizer  
        self.loss_fn = loss_fn
        self.decoder = MidiDecoder()
        self.sample_path = sample_path

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        log_dir = Path(log_dir).resolve() / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = Path(ckpt_dir).resolve() / timestamp
        ckpt_path.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir)

        self.checkpointer = StandardCheckpointHandler()

    def compile(self, rng, input_shape):
        params = self.model.init(rng, jax.numpy.ones(input_shape, dtype=jax.numpy.int32))
        self.opt_state = self.optimizer.init(params)
        self.params = params

    def train_step(self, batch, rng):
        inputs = batch  # batch = input_seq, targets created below
        targets = inputs[:, 1:]
        input_seq = inputs[:, :-1]
    
        def loss_fn(params):
            logits = self.model.apply(params, input_seq, rng=rng, train=True)
            # Assume logits shape: (batch, seq_len, vocab_size), targets: (batch, seq_len)
            return self.loss_fn(logits, targets)

        grads = jax.grad(loss_fn)(self.params)
        loss = loss_fn(self.params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)
        return loss

    def eval_step(self, batch):
        inputs = batch
        targets = inputs[:, 1:]
        input_seq = inputs[:, :-1]
        
        logits = self.model.apply(self.params, input_seq, train=False)
        loss = self.loss_fn(logits, targets)

        predictions = jax.numpy.argmax(logits, axis=-1)
        accuracy = jax.numpy.mean(predictions == targets)

        return loss, accuracy

    def train(self, epochs, batch_size=None):
        assert self.params is not None, "Model is not compiled. Please call compile() before train()."

        for epoch in range(epochs):
            rng = jax.random.PRNGKey(epoch)
            for batch, song_names in tqdm(self.dataloader.load_data(batch_size=batch_size, shuffle=True, key=rng), desc=f"Epoch {epoch+1}/{epochs}"):
                loss = self.train_step(batch, rng)

            self.writer.add_scalar('Loss/train', loss, self.global_step)

            if self.validation_dataloader and epoch % 4 == 0:
                val_loss = 0
                val_accuracy = 0
                for val_batch, val_song_names in self.validation_dataloader.load_data() :
                    batch_loss, batch_accuracy = self.eval_step(val_batch)
                    val_accuracy += batch_accuracy
                    val_loss += batch_loss
                val_loss /= len(self.validation_dataloader)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)

            # Save checkpoints periodically
            if epoch % 10 == 0:
                self.checkpointer.save(self.ckpt, self.params, step=epoch)

            # Generate a sample for each epoch
            # Use a fixed start token or random input from the batch for generation
            start_token = jax.numpy.array([[0]], dtype=jax.numpy.int32)  
            sample = self._generate_sample(start_token)
            self.decoder.decode(sample, output_path=self.sample_path / f"sample_epoch_{epoch}.mid")

    def _generate_sample(self, input_seq, max_length=10000):
        tokens = list(input_seq[0])  # start with the initial sequence (assume batch size 1)
        current_seq = input_seq
        for _ in range(max_length):
            logits = self.model.apply(self.params, current_seq)
            next_token = jax.numpy.argmax(logits[:, -1, :], axis=-1)  
            tokens.append(int(next_token[0]))
            current_seq = jax.numpy.concatenate([current_seq, next_token[:, None]], axis=1)
        return tokens