import os
import argparse
import torch
import json
import torchaudio
import evaluate
import datasets 
import datetime
import logging
import gc
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from peft import prepare_model_for_int8_training
from peft import LoraConfig, PeftModel, LoraModel, get_peft_model, PeftModel, PeftConfig
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
import numpy as np
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


class Load_Data(Dataset):
    def __init__(self, args, json_file, device):
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
        self.tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=args.language, task=args.task)
        self.data = []
        self.device = device
        with open(json_file, "r") as file:
            for line in file:
                try:
                    data = json.loads(line)
                    self.data.append(data)
                except json.JSONDecodeError as e:
                    print("Error decoding JSON:", e)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        audio_path = self.data[idx]['audio_filepath']
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_np_array = waveform.numpy()  
        input_features = self.feature_extractor(audio_np_array, sampling_rate=16000).input_features[0]
        
        transcription = self.data[idx]['text'] 
        labels = self.tokenizer(transcription).input_ids
        
        data_dict = {
            'input_features': input_features,
            'labels': labels
        }
        return data_dict
        
    def to(self, device):
        # 将数据移动到指定的设备上
        self.device = device
        return self

        

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def compute_metrics(pred, args):
    metric = evaluate.load("wer")
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=args.language, task=args.task)
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)
    
# This callback helps to save only the adapter weights and remove the base model weights.
class SavePeftModelCallback(TrainerCallback):
        def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            print(" SavePeftModelCallback True ")
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
    
            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)
    
            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control


def train_model(args):
    
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging_dir=os.path.join(args.output_dir, current_datetime, "logs")
    os.makedirs(logging_dir, exist_ok=True)
    print(logging_dir)

    if torch.cuda.is_available():
        # 使用第一个可用的GPU（通常是GPU 0）
        device = torch.device("cuda:0")
        print("GPU is available")
    else:
        # 如果没有GPU可用，使用CPU进行计算
        print("GPU is not available")
    
    train_dataset = Load_Data(args, args.manifest_filepath, device).to(device)
    validate_dataset = Load_Data(args, args.validate_filepath, device).to(device)
    
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.language, task=args.task)
    processor = processor
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    metric = evaluate.load("wer")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path, load_in_8bit=True)
    model = prepare_model_for_int8_training(model)
    
    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
    
    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    training_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(args.output_dir, current_datetime),  # change to a repo name of your choice
    per_device_train_batch_size=args.train_batchsize,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=args.lr,
    warmup_steps=50,
    num_train_epochs=args.epoch,
    evaluation_strategy="steps",
    fp16=True,
    per_device_eval_batch_size=args.validate_batchsize,
    generation_max_length=128,
    logging_dir=logging_dir,
    logging_strategy="steps",
    logging_first_step=True,
    logging_steps=100,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=10,
    eval_steps=1000,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
)
    trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validate_dataset,
    data_collator=data_collator,
    compute_metrics=lambda pred: compute_metrics(pred, args),
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],
)
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    trainer.save_model()

    access_token = "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"
    model.push_to_hub(repo_id=args.peft_model_id, create_pr=1, token=access_token, local_files=training_args.output_dir)
    
def test_model(args):
    if torch.cuda.is_available():
        # 使用第一个可用的GPU（通常是GPU 0）
        device = torch.device("cuda:0")
        print("GPU is available")
    else:
        # 如果没有GPU可用，使用CPU进行计算
        print("GPU is not available")
    
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.language, task=args.task)
    processor = processor
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    peft_config = PeftConfig.from_pretrained(pretrained_model_name_or_path=args.peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto")
    model = PeftModel.from_pretrained(model, args.peft_model_id)
    model.config.use_cache = True

    test_dataset = Load_Data(args, args.test_filepath, device)
    eval_dataloader = DataLoader(test_dataset, batch_size=args.test_batchsize, collate_fn=data_collator)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    normalizer = BasicTextNormalizer()
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    predictions = []
    references = []
    normalized_predictions = []
    normalized_references = []
    
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to("cuda"),
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend(decoded_labels)
                normalized_predictions.extend([normalizer(pred).strip() for pred in decoded_preds])
                normalized_references.extend([normalizer(label).strip() for label in decoded_labels])
            del generated_tokens, labels, batch
        gc.collect()
    wer = 100 * wer_metric.compute(predictions=predictions, references=references)
    normalized_wer = 100 * wer_metric.compute(predictions=normalized_predictions, references=normalized_references)
    cer = 100 * cer_metric.compute(predictions=predictions, references=references)
    normalized_cer = 100 * cer_metric.compute(predictions=normalized_predictions, references=normalized_references)
    eval_metrics = {"eval/wer": wer, "eval/normalized_wer": normalized_wer, "eval/cer": cer, "eval/normalized_cer": normalized_cer}
    
    # print(f"{wer=} and {normalized_wer=}")
    print(eval_metrics)
    logging.info(f'Finished testing. Test results: {eval_metrics}')
    

def main():
# /data/BEA-Base.json/dev-repet.json
# /data2/Users/yan/Dataset/CommonVoice/cv-corpus-15.0-2023-09-08/hu/test.json
# /data2/Users/yan/Dataset/CommonVoice/cv-corpus-12.0-2022-12-07/hu/test.json
# /data2/Users/yan/Dataset/CommonVoice/cv-corpus-7.0-2021-07-21/hu/test.json
# openai/whisper-large-v2  openai/whisper-large  openai/whisper-medium  openai/whisper-small  openai/whisper-base  openai/whisper-tiny
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="openai/whisper-large-v3", help='Input model name or file path')
    parser.add_argument('--language', default="hu", help='Language Type')
    parser.add_argument('--task', default="transcribe", help='Task Type')
    parser.add_argument('--train_batchsize', default=16, help='Input train batchsize')
    parser.add_argument('--validate_batchsize', default=1, help='Input validate batchsize')
    parser.add_argument('--test_batchsize', default=1, help='Input test batchsize')
    parser.add_argument('--output_dir', default="/data2/Users/yan/01Results/whisper/fast-whisper-finetuning/lora_result/large-v3/", help='output filepath')

    parser.add_argument('--train_or_test', default="train", help='train or test')
    parser.add_argument('--manifest_filepath', default="/data/BEA-Base.json/train-114.json", help='Input *.json train dataset file path')
    parser.add_argument('--validate_filepath', default="/data/BEA-Base.json/dev-spont.json", help='Input *.json validation dataset file path')
    parser.add_argument('--test_filepath', default="/data2/Users/yan/Dataset/CommonVoice/cv-corpus-12.0-2022-12-07/hu/test.json", help='Input *.json test dataset file path')
    parser.add_argument('--lr', default=1e-3, help='learning rate')
    parser.add_argument('--epoch', default=5, help='Epoch')
    parser.add_argument('--peft_model_id', default="Yan20231017/whisper-large-v3-lora-hu-epoch5", help='peft_model_id')
    args = parser.parse_args()

    if args.train_or_test == "train":
        train_model(args)
    else:
        test_model(args)

    

if __name__ == '__main__':
    main()
    
