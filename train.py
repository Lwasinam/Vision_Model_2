from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
import wandb
import accelerate
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import load_model, save_model
from accelerate import Accelerator
from transformers import GPT2TokenizerFast
import threading


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.convert_tokens_to_ids('[SOS]')
    eos_idx = tokenizer_tgt.convert_tokens_to_ids('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.module.encode(source, None)
   
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).long().to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).long().to(device)
      

        # calculate output
        out = model.module.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        # print(f'out: {out.shape}')

        # Get next token probabilities with temperature applied
        logits = model.module.project(out[:, -1]) 
        probabilities = F.softmax(logits, dim=-1)

        # Greedily select the next word
        next_word = torch.argmax(probabilities, dim=1)
        
        # Append next word
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)
        # # get next token
        # prob = model.project(out[:, -1])
        # _, next_word = torch.max(prob, dim=1)
        # # print(f'prob: {prob.shape}')
        # decoder_input = torch.cat(
        #     [decoder_input, torch.empty(1, 1).long().fill_(next_word.item()).to(device)], dim=1
        # )

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds,tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=3):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)+_
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, None, tokenizer_tgt, max_len, device)

            # source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
           
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            # print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    # if writer:
    #     # Evaluate the character error rate
    #     # Compute the char error rate 
    #     metric = torchmetrics.CharErrorRate()
    #     cer = metric(predicted, expected)
    #     writer.add_scalar('validation cer', cer, global_step)
    #     writer.flush()

    #     # Compute the word error rate
    #     metric = torchmetrics.WordErrorRate()
    #     wer = metric(predicted, expected)
    #     writer.add_scalar('validation wer', wer, global_step)
    #     writer.flush()

    #     # Compute the BLEU metric
    #     metric = torchmetrics.BLEUScore()
    #     bleu = metric(predicted, expected)
    #     writer.add_scalar('validation BLEU', bleu, global_step)
    #     writer.flush()

def get_all_sentences(ds):
    for item in ds:
        yield item['text']
def batch_iterator(data):
    for i in range(0, len(data)):
        yield data[i]['text'] 

# Assuming batch_iterator is a function that yields batches
def tqdm_batch_iterator(data, *args, **kwargs):
    for batch in tqdm(batch_iterator(data, *args, **kwargs), total=len(data)):
        yield batch               

def get_or_build_tokenizer(config, ds):
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2", unk_token ='[UNK]', bos_token = '[SOS]', eos_token = '[EOS]' , pad_token = '[PAD]')
    return tokenizer
    # tokenizer_path = Path(config['tokenizer_file'])
    # if not Path.exists(tokenizer_path):
    #     # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
    #     tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    #     tokenizer.pre_tokenizer = Whitespace()
    #     trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
    #     tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
    #     tokenizer.save(str(tokenizer_path))
    # else:
    #     tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    # ds_raw = load_dataset("HausaNLP/HausaVG", split='train+validation+test+challenge_test')
    train_ds_raw =  load_dataset("priyank-m/MJSynth_text_recognition", split='train')
    
    val_ds_raw =  load_dataset("priyank-m/MJSynth_text_recognition", split='train')
    
    # ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    
    tokenizer_tgt = get_or_build_tokenizer(config, train_ds_raw,)
    seed = 20  # You can choose any integer as your seed
    torch.manual_seed(seed)
    # # Keep 90% for training, 10% for validation
    # train_ds_size = int(0.9 * len(ds_raw))
    # val_ds_size = len(ds_raw) - train_ds_size
    # train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_tgt,  config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_tgt, config['seq_len'])
    

    train_dataloader = DataLoader(train_ds,batch_size=config['batch_size'], shuffle=True )
   
    val_dataloader = DataLoader(val_ds, batch_size=1,shuffle=True )

    return train_dataloader, val_dataloader, tokenizer_tgt

def get_model(config, vocab_tgt_len):
    model = build_transformer(vocab_tgt_len, config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):

    accelerator = Accelerator(mixed_precision='fp16')
  


    wandb.login(key = 'c20a1022142595d7d1324fdc53b3ccb34c0ded22')
    wandb.init(project="Vision", name=config['project_name'])

    # Initialize WandB configuration
    wandb.config.epochs = config['num_epochs']
    wandb.config.batch_size = config['batch_size']
    wandb.config.learning_rate = config['lr'] 
    # Define the devic
    # Define the device
    device = accelerator.device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Make sure the weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_tgt = get_ds(config)
    model = get_model(config, len(tokenizer_tgt)).to(device)
   

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98),eps=1e-9)

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader
)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0

    def save_models():
        accelerator.save_state(output_dir=f'/kaggle/working/weights/tmodel_00')
        print(f'saving global step {global_step}')
    
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        accelerator.load_state(model_filename)
        initial_epoch = 1
   
        # state = torch.load(model_filename)
        # model.load_state_dict(state['model_state_dict'])
        # initial_epoch = state['epoch'] + 1
        # optimizer.load_state_dict(state['optimizer_state_dict'])
        # global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.convert_tokens_to_ids('[PAD]'), label_smoothing=0.1).to(device)
   
    for epoch in range(initial_epoch, config['num_epochs']):

        # timer = threading.Timer(5*60, save_models)
        # timer.start()

        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        
        for batch in batch_iterator:
            run_validation(model, val_dataloader, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step)
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (B, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.module.encode(encoder_input, None) # (B, seq_len, d_model)
            decoder_output = model.module.decode(encoder_output, None, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.module.project(decoder_output)
            
             # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch["label"].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, len(tokenizer_tgt)), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            wandb.log({"Training Loss": loss.item(), "Global Step": global_step})

            # # Backpropagate the loss
            # loss.backward()
            accelerator.backward(loss)

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # # Run validation at the end of every epoch
            # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'global_step': global_step
        # }, model_filename)
        # accelerator.save_model(model, model_filename)
    
        accelerator.save_state(output_dir=f'/kaggle/working/weights/tmodel_{epoch:02d}')
        # run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        model.eval()
        eval_loss = 0.0

        #accelerate 
        accurate = 0
        num_elems = 0
        # batch_iterator = tqdm(v_dataloader, desc=f"Processing Epoch {epoch:02d}")
        with torch.no_grad():
            batch_itere = tqdm(val_dataloader, desc=f"Processing loss")
            for batch in batch_itere:
            

                encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
                decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
                encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
                decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

                # Run the tensors through the encoder, decoder and the projection layer
            
                encoder_output = model.module.encode(encoder_input, None) # (B, seq_len, d_model)
                decoder_output = model.module.decode(encoder_output, None, decoder_input, decoder_mask)# (B, seq_len, d_model)
                proj_output = model.module.project(decoder_output)
            
                # (B, seq_len, vocab_size)

                # Compare the output with the label
                # label = batch['label'].to(device) # (B, seq_len)
                proj_output, label = accelerator.gather_for_metrics((
                proj_output, batch["label"]
            ))

                # Compute the loss using a simple cross entropy
                ls = loss_fn(proj_output.view(-1, len(tokenizer_tgt)), label.view(-1))
                batch_itere.set_postfix({"loss": f"{ls.item():6.3f}"})
                eval_loss += ls
                # loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
           
                
        avg_val_loss = eval_loss / len(val_dataloader)
        accelerator.print(f"Epoch {epoch},Validation Loss: {avg_val_loss})Validation Loss: {avg_val_loss}")
        # print(f'Epoch {epoch},Validation Loss: {avg_val_loss.item()}')
        wandb.log({"Validation Loss": avg_val_loss.item(), "Global Step": global_step})

    
        run_validation(model, val_dataloader, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
