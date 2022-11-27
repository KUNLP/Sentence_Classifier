import argparse
import os
import logging
from attrdict import AttrDict

import torch

# KorSciBERT
import transformers
from KorSciBERT.korscibert_v1_tf.tokenization_kisti import FullTokenizer

# huggingface/lm
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import ElectraModel, RobertaModel, BertModel, AutoModel, AlbertModel


from src.model.model_graph_cls import LMForSequenceClassification, LMForSequenceClassification_multi_loss, \
    LMForSequenceClassification_more_layer, LMForSequenceClassification_maxpooling, \
    LMForSequenceClassification_only_LabelAttn, LMForSequenceClassification_without_LabelAttn, \
    LMForSequenceClassification_only_HierarchicalAttn, Baseline, LMForSequenceClassification_softmax
# from src.model.model_graph import LMForSequenceClassification, LMForSequenceClassification1, LMForSequenceClassification2

from src.model.main_functions_graph import train, evaluate, predict

from src.functions.utils import init_logger, set_seed

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

def create_model(args):

    if "KorSciBERT" in args.output_dir:
        # 모델 파라미터 Load
        config = transformers.BertConfig.from_pretrained(
            './KorSciBERT/model/bert_config_kisti.json',
            # if args.from_init_weight else os.path.join(args.output_dir,"model/checkpoint-{}/config.json".format(args.checkpoint)),
            # cache_dir=args.cache_dir,
        )
        config.vocab_size = 15330

    else:
        # 모델 파라미터 Load
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            # if args.from_init_weight else os.path.join(args.output_dir, "model/checkpoint-{}".format(args.checkpoint)),
            # cache_dir=args.cache_dir,
        )

    config.num_labels = 9
    # attention 추출하기
    config.output_attentions=True
    # print(config)

    if "KorSciBERT" in args.output_dir:
        tokenizer = FullTokenizer(
            vocab_file="./KorSciBERT/korscibert_v1_tf/vocab_kisti.txt",
            do_lower_case=args.do_lower_case,
            tokenizer_type="Mecab"
        )

        tokenizer.cls_token_id = 5
        tokenizer.sep_token_id = 6
        tokenizer.pad_token_id = 3

        tokenizer.padding_side = "right"

    else:
        # tokenizer는 pre-trained된 것을 불러오는 과정이 아닌 불러오는 모델의 vocab 등을 Load
        # BertTokenizerFast로 되어있음
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            # if args.from_init_weight else os.path.join(args.output_dir,"model/checkpoint-{}".format(args.checkpoint)),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir,
        )

    print(tokenizer)

    if "roberta" in args.output_dir:
        language_model = AutoModel.from_pretrained(
            args.model_name_or_path,
            config=config)

    elif "bert" in args.output_dir:
        language_model = AutoModel.from_pretrained(
            args.model_name_or_path,
            config=config)

    elif "KorSciBERT" in args.output_dir:
        language_model = BertModel.from_pretrained(
            './KorSciBERT/pytorch_model.bin',
            config=config)

    # model = Baseline(
    model = LMForSequenceClassification( # proposed model
    # model = LMForSequenceClassification_softmax(
    # model = LMForSequenceClassification_only_HierarchicalAttn(
    # model = LMForSequenceClassification_only_LabelAttn(
    # model = LMForSequenceClassification_without_LabelAttn(
        config=config,
        max_sentence_length=args.max_sentence_length,
        language_model=language_model,
        device=args.device,
        d_embed=args.d_embed,
        graph_dim=args.d_embed, gcn_dep=args.gcn_dep, gcn_layers=args.gcn_layers,
        coarse_add=args.coarse_add,
        # from_tf=True,
    )

    if not args.from_init_weight: model.load_state_dict(torch.load(os.path.join(args.output_dir, "model/checkpoint-{}/pytorch.pt".format(args.checkpoint))))
    # print(model)

    model.to(args.device)
    return model, tokenizer

def main(cli_args):
    # 파라미터 업데이트
    args = AttrDict(vars(cli_args))
    args.device = "cuda"
    logger = logging.getLogger(__name__)

    if "no_coarse" in args.output_dir:
        args.coarse_add = False
    if "HierarchicalAttn" in args.output_dir:
        args.coarse_add = False
    if "wo_LabelAttn" in args.output_dir:
        args.coarse_add = False
    if "baseline" in args.output_dir:
        args.coarse_add = False


    # logger 및 seed 지정
    init_logger()
    set_seed(args)

    # 모델 불러오기
    print(args.output_dir)
    model, tokenizer = create_model(args)

    # Running mode에 따른 실행
    if args.do_train:
        train(args, model, tokenizer, logger)
    elif args.do_eval:
        evaluate(args, model, tokenizer, logger)
    elif args.do_predict:
        print(True)
        predict(args, model, tokenizer)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    # Directory

    #------------------------------------------------------------------------------------------------
    # cli_parser.add_argument("--data_dir", type=str, default="./data")
    cli_parser.add_argument("--data_dir", type=str, default="./data/origin/merge_origin_preprocess")


    # # word unit
    # cli_parser.add_argument("--train_file", type=str, default='new_train.json')
    # cli_parser.add_argument("--eval_file", type=str, default='new_test.json')
    # cli_parser.add_argument("--predict_file", type=str, default='new_test.json')

    # chunk unit
    cli_parser.add_argument("--train_file", type=str, default='origin_train.json')
    cli_parser.add_argument("--eval_file", type=str, default='origin_test.json')
    cli_parser.add_argument("--predict_file", type=str, default='origin_test.json')

    # ------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------

    # KorSciBERT
    cli_parser.add_argument("--model_name_or_path", type=str, default='./KorSciBERT/model')
    cli_parser.add_argument("--cache_dir", type=str, default='./KorSciBERT/model')
    cli_parser.add_argument("--output_dir", type=str, default="./KorSciBERT/cls/graph_concat_coarse_alpha") # acc: 89.94, F1: 90.22
   
    cli_parser.add_argument("--max_sentence_length", type=int, default=110)

    # https://github.com/KLUE-benchmark/KLUE-baseline/blob/main/run_all.sh
    # Model Hyper Parameter
    cli_parser.add_argument("--max_seq_length", type=int, default=512)
    cli_parser.add_argument("--d_embed", type=int, default=384)  # 3*128=384  # num_multi_layers의 배수
    cli_parser.add_argument("--gcn_dep", type=float, default=0.0)  # 0.1)
    cli_parser.add_argument("--gcn_layers", type=int, default=3)

    # Training Parameter
    cli_parser.add_argument("--learning_rate", type=float, default=2e-5)
    cli_parser.add_argument("--train_batch_size", type=int, default=16)
    cli_parser.add_argument("--eval_batch_size", type=int, default=16)
    cli_parser.add_argument("--num_train_epochs", type=int, default=5)

    #cli_parser.add_argument("--save_steps", type=int, default=2000)
    cli_parser.add_argument("--logging_steps", type=int, default=100)
    cli_parser.add_argument("--seed", type=int, default=42)
    cli_parser.add_argument("--threads", type=int, default=8)

    cli_parser.add_argument("--weight_decay", type=float, default=0.0)
    cli_parser.add_argument("--adam_epsilon", type=int, default=1e-10)
    cli_parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    cli_parser.add_argument("--warmup_steps", type=int, default=0)
    cli_parser.add_argument("--max_steps", type=int, default=-1)
    cli_parser.add_argument("--max_grad_norm", type=int, default=1.0)


    cli_parser.add_argument("--verbose_logging", type=bool, default=False)
    cli_parser.add_argument("--do_lower_case", type=bool, default=False)
    cli_parser.add_argument("--no_cuda", type=bool, default=False)

    # Running Mode
    cli_parser.add_argument("--from_init_weight", type=bool, default= True) #False) #True)
    cli_parser.add_argument("--checkpoint", type=str, default="4")
    cli_parser.add_argument("--coarse_add", type=bool, default=True)

    cli_parser.add_argument("--do_train", type=bool, default= True) #False) #True)
    cli_parser.add_argument("--do_eval", type=bool, default=False)
    cli_parser.add_argument("--do_predict", type=bool, default= False) #True) #False)

    cli_args = cli_parser.parse_args()

    main(cli_args)

