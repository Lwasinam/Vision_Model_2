import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from transformers import ViTFeatureExtractor
from io import BytesIO
from base64 import b64decode
from PIL import Image
import base64

# import model
model_id = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(
    model_id
)
class BilingualDataset(IterableDataset):

    def __init__(self, ds,tokenizer_tgt, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_tgt = tokenizer_tgt
        # self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.convert_tokens_to_ids("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.convert_tokens_to_ids("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.convert_tokens_to_ids("[PAD]")], dtype=torch.int64)

    # def __len__(self):
    #     return 1200000
    # def __getitem__(self):
    #     pass
  
    def generate(self):
        for item in self.ds:

           
            src_image = item['jpg']
            tgt_text = item['txt']

            # base64_bytes = base64.b64encode(src_image)
        

            # src_image = base64_bytes.decode()
            

            # src_image  = Image.open(BytesIO(b64decode(''.join(src_image))))

            if src_image.mode != 'RGB':
                src_image = src_image.convert('RGB')
                
            enc_input = feature_extractor(
            src_image,
            return_tensors='pt'
            )
                

                # Transform the text into tokens
            dec_input_tokens = self.tokenizer_tgt.encode(tgt_text)

                # # Add sos, eos and padding to each sentence
                # enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
                # We will only add <s>, and </s> only on the label
            dec_input_tokens = dec_input_tokens[:self.seq_len-1]
            dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) -1

                # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
            if dec_num_padding_tokens < 0:
                raise ValueError("Sentence is too long")

                # # Add <s> and </s> token
                # encoder_input = torch.cat(
                #     [
                #         self.sos_token,
                #         torch.tensor(enc_input_tokens, dtype=torch.int64),
                #         self.eos_token,
                #         torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
                #     ],
                #     dim=0,
                # )

                # Add only <s> token
            decoder_input = torch.cat(
                    [
                        self.sos_token,
                        torch.tensor(dec_input_tokens, dtype=torch.int64),
                        torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
                    ],
                    dim=0,
                )

                # Add only </s> token
            label = torch.cat(
                    [
                        torch.tensor(dec_input_tokens, dtype=torch.int64),
                        self.eos_token,
                        torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
                    ],
                    dim=0,
                )


            assert decoder_input.size(0) == self.seq_len
            assert label.size(0) == self.seq_len
           
             
            encoder_input = enc_input['pixel_values'][0]  # (seq_len)
            decoder_input = decoder_input  # (seq_len)
            encoder_mask = (torch.cat((torch.ones(197,),torch.zeros(63),),)).unsqueeze(0).unsqueeze(0) # (1, 1, seq_len)
            decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)) # (1, seq_len) & (1, seq_len, seq_len),
            label = label  
                    # "src_text": src_text,
            tgt_text = tgt_text
            yield encoder_input, decoder_input, encoder_mask, decoder_mask, label
    def __iter__(self):
        return iter(self.generate())           
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0