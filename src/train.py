from model.selenite import Selenite
import optax

import jax
import jax.numpy as jnp
from music_utils.midi.midi_decoder import MidiDecoder
from pathlib import Path
from music_utils.piano_aria.piano_aria_dataloader import PianoAriaDataloader
from params import INPUT_PATH, MAX_SEQUENCE_LENGTH, OUTPUT_PATH
from model.trainer import Trainer
from music_utils.midi.data_split_utils import split_files

def __main__():
    
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")

    train_files, test_files = split_files(folder_path=INPUT_PATH, train_size=0.8, test_size=0.2)

    train_dataloader = PianoAriaDataloader(files=train_files)
    test_dataloader = PianoAriaDataloader(files=test_files)

    model = Selenite(
        vocab_size=train_dataloader.vocab_size(),
        d_model=128,
        d_ff=512,
        n_heads=4,
        n_layers=2
    )
    

    trainer = Trainer(
        model=model,
        dataloader=train_dataloader,
        validation_dataloader=test_dataloader,
        optimizer=optax.adam(learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8),
        loss_fn= lambda logits, targets: jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets)),
        sample_path=OUTPUT_PATH + "/samples",
        log_dir=OUTPUT_PATH + "/logs",
        ckpt_dir=OUTPUT_PATH + "/checkpoints"
    )
    trainer.compile(rng=jax.random.PRNGKey(0), input_shape=(1, MAX_SEQUENCE_LENGTH))

    trainer.train(epochs=20, batch_size=8)

if __name__ == "__main__":
    __main__()
