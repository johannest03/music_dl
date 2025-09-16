from datetime import datetime
from pathlib import Path
import jax
from torch.utils.tensorboard import SummaryWriter
from music_utils.midi.midi_decoder import MidiDecoder
from music_utils.midi.midi_token_id_conversion import MidiTokenIDConversion
import orbax.checkpoint as ocp
import optax
from params import OUTPUT_PATH
from tqdm import tqdm
from functools import partial

class Trainer():

    def __init__(self, model, dataloader, validation_dataloader, optimizer, loss_fn, log_dir, ckpt_dir, sample_dir, load_ckpt_path=None, timestamp=None):
        # Model and training components
        # Provide load_ckpt_path to resume training from a checkpoint, to keep same directory structure use same timestamp (directory name)
        
        self.model = model
        self.dataloader = dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optimizer  
        self.loss_fn = loss_fn
        self.decoder = MidiDecoder()
        self.midi_token_id_conversion = MidiTokenIDConversion()
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.sample_dir = Path(sample_dir).resolve() / timestamp
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(log_dir).resolve() / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = ocp.test_utils.erase_and_create_empty(Path(ckpt_dir).resolve() / timestamp)

        self.writer = SummaryWriter(self.log_dir)

        self.checkpointer = ocp.StandardCheckpointer()
        
        self.load_ckpt_path = load_ckpt_path
        
       

    def compile(self, rng, input_shape):
        if self.load_ckpt_path: # load checkpoint if provided
            self.params = self.checkpointer.restore(self.load_ckpt_path)
            self.opt_state = self.optimizer.init(self.params)
            print(f"Loaded checkpoint from {self.load_ckpt_path}")
            return
        params = self.model.init(rng, jax.numpy.ones(input_shape, dtype=jax.numpy.int32))
        self.opt_state = self.optimizer.init(params)
        self.params = params
        
    @staticmethod
    @partial(jax.jit, static_argnames=['model', 'loss_fn', 'optimizer', 'midi_token_id_conversion'])
    def train_step(params, opt_state, model, batch, rng, loss_fn, optimizer, midi_token_id_conversion):
        inputs = batch
        targets = inputs[:, 1:]
        input_seq = inputs[:, :-1]

        def dloss_fn(params):
            logits = model.apply(params, input_seq, rng=rng, train=True)
            note_ids = jax.numpy.array(midi_token_id_conversion.get_note_ids())  # Convert to JAX array
            mask = jax.numpy.ones_like(targets, dtype=jax.numpy.float32) # penalize notes more heavily
            mask = jax.numpy.where(jax.numpy.isin(targets, note_ids), 1.0, 1.0)  # 1.0 for note tokens, 1.0 for others
            loss = loss_fn(logits, targets)
            loss = jax.numpy.mean(loss * mask)  # penalize notes more heavily
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(dloss_fn, has_aux=True)(params)
        acc = jax.numpy.mean(jax.numpy.argmax(logits, axis=-1) == targets)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, acc, params, opt_state
    
    @staticmethod
    @partial(jax.jit, static_argnames=['model', 'loss_fn'])
    def eval_step(params, model, loss_fn, batch, rng):
        inputs = batch
        targets = inputs[:, 1:]
        input_seq = inputs[:, :-1]

        logits = model.apply(params, input_seq, rng=rng, train=False)
        loss = loss_fn(logits, targets)

        predictions = jax.numpy.argmax(logits, axis=-1)
        accuracy = jax.numpy.mean(predictions == targets)

        return loss, accuracy, logits

    def train(self, epochs, batch_size=None):
        assert self.params is not None, "Model is not compiled. Please call compile() before train()."

        for epoch in range(epochs):
            rng = jax.random.PRNGKey(epoch)
            total_batches = self.dataloader.length // batch_size if batch_size else 1
            train_loss = 0
            train_acc = 0
            with tqdm(self.dataloader.load_data(batch_size=batch_size, shuffle=True, key=rng), desc=f"Epoch {epoch+1}/{epochs}", total=total_batches) as pbar:
                for batch, song_names in pbar:
                    loss, acc, self.params, self.opt_state = self.train_step(self.params, self.opt_state, self.model, batch, rng, self.loss_fn, self.optimizer, self.midi_token_id_conversion)
                    train_loss += loss
                    train_acc += acc
                    pbar.set_postfix({'loss': float(train_loss / (pbar.n + 1)), 'accuracy': float(train_acc / (pbar.n + 1))})

            self.writer.add_scalar('Loss/train', float(train_loss / total_batches), epoch)
            self.writer.add_scalar('Acc/train', float(train_acc / total_batches), epoch)

            if self.validation_dataloader and epoch % 4 == 0:
                val_loss = 0
                val_accuracy = 0
                val_perplexity = 0
                for val_batch, val_song_names in self.validation_dataloader.load_data():
                    batch_loss, batch_accuracy, logits = self.eval_step(self.params, self.model, self.loss_fn, val_batch, rng=rng)
                    val_accuracy += batch_accuracy
                    val_loss += batch_loss
                    targets = val_batch[:, 1:]
                    val_perplexity += self._compute_perplexity(logits, targets)
                val_loss /= self.validation_dataloader.length // batch_size
                val_accuracy /= self.validation_dataloader.length // batch_size
                val_perplexity /= self.validation_dataloader.length // batch_size
                self.writer.add_scalar('Loss/val', float(val_loss), epoch)
                self.writer.add_scalar('Accuracy/val', float(val_accuracy), epoch)
                self.writer.add_scalar('Perplexity/val', float(val_perplexity), epoch)

            # Save checkpoints periodically
            if epoch % 2 == 0:
                self.checkpointer.save(self.ckpt_dir / f"epoch_{epoch}", self.params)

            # Generate a sample for each epoch
            # Use a fixed start token or random input from the batch for generation
            if self.decoder and self.midi_token_id_conversion:
                try:
                    start_token = jax.numpy.array([self.midi_token_id_conversion.start_token_id(), self.midi_token_id_conversion.track_start_token_id()], dtype=jax.numpy.int32)
                    sample = self._generate_sample(start_token, rng=rng)
                    self.decoder.decode(sample, output_path=self.sample_dir / f"sample_epoch_{epoch}.mid")
                except Exception as e:
                    print(f"Sample generation failed at epoch {epoch}: {e}")

    def _compute_perplexity(self, logits, targets):
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_log_probs = jax.numpy.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
        perplexity = jax.numpy.exp(-jax.numpy.mean(target_log_probs))
        return perplexity

    def _generate_sample(self, input_seq, rng, max_length=200, temperature=1.0, top_k=50):
        tokens = list(input_seq)
        seq = jax.numpy.full((1, max_length), fill_value=self.midi_token_id_conversion.pad_token_id(), dtype=jax.numpy.int32)
        seq = seq.at[0, :len(tokens)].set(jax.numpy.array(tokens, dtype=jax.numpy.int32))
        cur_len = len(tokens)
        pad_id = self.midi_token_id_conversion.pad_token_id()
        for i in tqdm(range(cur_len, max_length), "Generating sample..."):
            logits = self.model.apply(self.params, seq, rng=rng)
            # Apply temperature and top-k, exclude PAD
            scaled_logits = logits[:, i-1, :] / temperature
            # Set PAD logit to -inf to avoid it
            scaled_logits = scaled_logits.at[0, pad_id].set(-float('inf'))
            top_k_logits, top_k_indices = jax.lax.top_k(scaled_logits, top_k)
            probs = jax.nn.softmax(top_k_logits)
            next_token_idx = jax.random.categorical(rng, probs[0])
            next_token = top_k_indices[0, next_token_idx]
            seq = seq.at[0, i].set(next_token)
            tokens.append(int(next_token))
        
        print(self.midi_token_id_conversion.ids_to_tokens(tokens))
        return tokens