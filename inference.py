import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from transformers import ViTFeatureExtractor
from io import BytesIO
from base64 import b64decode
from PIL import Image
from accelerate import Accelerator
import base64
from config import get_config
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from model import build_transformer
import torch.nn.functional as F

def process(model,image, tokenizer, device):
    image = get_image(image)
    model.eval()
    with torch.no_grad():
        encoder_input = image.unsqueeze(0).to(device) # (b, seq_len)
        # decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
        # encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
        # decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

        model_out = greedy_decode(model, encoder_input, None, tokenizer, 261,device)
        model_text  = tokenizer.decode(model_out.detach().cpu().numpy())
        print(model_text)





       



# get image prompt 
def get_image(image):
# import model
    model_id = 'google/vit-base-patch16-224-in21k'
    feature_extractor = ViTFeatureExtractor.from_pretrained(
    model_id
    )

    
    image  = Image.open(BytesIO(b64decode(''.join(image))))

    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    enc_input = feature_extractor(
    image,
    return_tensors='pt'
    )

    return enc_input['pixel_values'][0]

    


#get tokenizer
def get_or_build_tokenizer(config):
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


# get model
def get_model(config, vocab_tgt_len):
    model = build_transformer(vocab_tgt_len, config['seq_len'], d_model=config['d_model'])
    return model

# greedy decode 
def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, None)
   
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).long().to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).long().to(device)
      

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        # print(f'out: {out.shape}')

        # Get next token probabilities with temperature applied
        logits = model.project(out[:, -1]) 
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

def image_base64():
    

    with open('5.jpg', 'rb') as image_file:
        base64_bytes = base64.b64encode(image_file.read())
       

        base64_string = base64_bytes.decode()
        return base64_string


def start():
    accelerator = Accelerator()
    device = accelerator.device

    config = get_config()
    tokenizer = get_or_build_tokenizer(config)
    model = get_model(config, tokenizer.get_vocab_size())
    model = accelerator.prepare(model)
    accelerator.load_state('C:/AI/projects/Vision_Model_2/models/weights/tmodel_03')

    image = image_base64()
   
    

    process(model, image, tokenizer, device)

start()

