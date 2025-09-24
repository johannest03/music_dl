from model.selenite import Selenite
import optax
import jax
import jax.numpy as jnp
from music_utils.piano_aria.piano_aria_dataloader import PianoAriaDataloader
from params import INPUT_PATH, MAX_SEQUENCE_LENGTH, OUTPUT_PATH, vocab_size
from model.trainer import Trainer
from music_utils.midi.data_split_utils import split_files

def loss_fn(logits, targets, pad_token_id):
    mask = targets != pad_token_id
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)

def __main__():
    
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")

    train_files, test_files = split_files(folder_path=INPUT_PATH, train_size=0.8, test_size=0.2)

    train_dataloader = PianoAriaDataloader(files=train_files)
    test_dataloader = PianoAriaDataloader(files=test_files)

    model = Selenite(
        vocab_size=vocab_size,
        pad_token_id=train_dataloader.tokenizer.pad_token_id
    )

    trainer = Trainer(
        model=model,
        dataloader=train_dataloader,
        validation_dataloader=test_dataloader,
        optimizer=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=1e-4)), 
        loss_fn=lambda logits, targets: loss_fn(logits, targets, pad_token_id=train_dataloader.tokenizer.pad_token_id),
        sample_dir=OUTPUT_PATH + "/samples",
        log_dir=OUTPUT_PATH + "/logs",
        ckpt_dir=OUTPUT_PATH + "/checkpoints"
    )
    trainer.compile(rng=jax.random.PRNGKey(0), input_shape=(1, MAX_SEQUENCE_LENGTH))

    trainer.train(epochs=100, batch_size=32)

if __name__ == "__main__":
    __main__()
