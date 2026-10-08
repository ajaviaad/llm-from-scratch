"""Command-line entry points for the from-scratch language-model workshop."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import numpy as np
from . import __version__
from .model import Config, TinyDecoder, Adam, read_documents, train_steps, evaluation_loss, save_checkpoint, load_checkpoint, generate
from .tokenization import ByteTokenizer, BPETokenizer, tokenizer_from_dict

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(value):
    number=int(value)
    if number<1: raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def output_json(value,path=None):
    text=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if path is None: print(text,end='')
    else: Path(path).write_text(text,encoding='utf-8')


def new_directory(path):
    path=path.expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('output directory must be new or empty; existing runs are preserved')
    path.mkdir(parents=True,exist_ok=True)
    return path


def tokenizer_for(kind,path,vocabulary):
    if kind=='byte': return ByteTokenizer()
    documents=[s for s in Path(path).read_text(encoding='utf-8').split('\n\n') if s.strip()]
    return BPETokenizer.train(documents,vocab_size=vocabulary)


def checkpoint_tokenizer(metadata):
    if metadata.get('tokenizer') is None: raise ValueError('checkpoint has no text-tokenizer metadata')
    return tokenizer_from_dict(metadata['tokenizer'])


def command_train(args):
    fingerprints={'train':digest(args.train_file),'validation':digest(args.validation_file)}
    if fingerprints['train']==fingerprints['validation']:
        raise ValueError('training and validation files must contain different documents')
    overrides=('context','width','heads','layers','feed_forward','seed','learning_rate','tokenizer','vocab_size')
    if args.resume:
        if any(getattr(args,key) is not None for key in overrides):
            raise ValueError('resume restores configuration, tokenizer, optimizer, and RNG; omit fresh-run options')
        model,optimizer,rng,metadata=load_checkpoint(args.resume)
        if metadata['data_digests']!=fingerprints: raise ValueError('resume data does not match checkpoint digests')
        tokenizer=checkpoint_tokenizer(metadata)
    else:
        tokenizer=tokenizer_for(args.tokenizer or 'byte',args.train_file,args.vocab_size or 320)
        default=asdict(Config())
        for name in ('context','width','heads','layers','feed_forward'):
            if getattr(args,name) is not None: default[name]=getattr(args,name)
        default['vocabulary']=tokenizer.vocab_size
        model=TinyDecoder(Config(**default),seed=17 if args.seed is None else args.seed)
        optimizer=Adam(model.parameters,.003 if args.learning_rate is None else args.learning_rate)
        rng=np.random.default_rng((17 if args.seed is None else args.seed)+1)
    train=read_documents(args.train_file,model.config.context,tokenizer)
    validation=read_documents(args.validation_file,model.config.context,tokenizer)
    train_texts={s.strip() for s in args.train_file.read_text(encoding='utf-8').split('\n\n') if s.strip()}
    valid_texts={s.strip() for s in args.validation_file.read_text(encoding='utf-8').split('\n\n') if s.strip()}
    if train_texts & valid_texts: raise ValueError('an exact document appears in both train and validation')
    output=new_directory(args.output)
    initial=evaluation_loss(model,validation); start=optimizer.step
    history=train_steps(model,optimizer,train,rng,args.steps)
    final=evaluation_loss(model,validation)
    save_checkpoint(output,model,optimizer,rng,fingerprints,tokenizer)
    report={'result_class':'trained-small-language-model',
            'repository_version':__version__,'python':platform.python_version(),'numpy':np.__version__,
            'config':asdict(model.config),'parameter_count':sum(p.size for p in model.parameters.values()),
            'tokenizer_kind':tokenizer.kind,'tokenizer_sha256':tokenizer.digest(),
            'data_digests':fingerprints,'training_documents':len(train),'validation_documents':len(validation),
            'sampling':'uniform document then uniform valid start, batch 4; windows never cross document boundaries',
            'validation_protocol':'8 deterministic batches, seed 2026; same windows before and after updates',
            'start_step':start,'end_step':optimizer.step,'additional_steps':args.steps,
            'initial_validation_nll':initial,'final_validation_nll':final,
            'final_validation_perplexity':math.exp(final),'history':history,
            'interpretation':'Real parameter training on the identified corpus; not evidence of general language, reasoning, or assistant capability.'}
    output_json(report,output/'training.json')
    output_json({'status':'trained','output':str(output),'steps':optimizer.step,
                 'initial_validation_nll':initial,'final_validation_nll':final,'parameters':report['parameter_count']})


def build_parser():
    parser=argparse.ArgumentParser(description='Build and train a decoder language model from scratch using NumPy.')
    parser.add_argument('--version',action='version',version=__version__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('check',help='Run the small numerical reference checks')
    tokenize=sub.add_parser('tokenize',help='Inspect byte or train-only byte-pair tokenization')
    tokenize.add_argument('--text',required=True)
    tokenize.add_argument('--tokenizer',choices=['byte','bpe'],default='byte')
    tokenize.add_argument('--train-file',type=Path,default=ROOT/'data/train.txt')
    tokenize.add_argument('--vocab-size',type=positive,default=320)
    train=sub.add_parser('train',help='Train all parameters, or resume a complete checkpoint')
    train.add_argument('--output',type=Path,required=True)
    train.add_argument('--steps',type=positive,default=120)
    train.add_argument('--train-file',type=Path,default=ROOT/'data/train.txt')
    train.add_argument('--validation-file',type=Path,default=ROOT/'data/validation.txt')
    train.add_argument('--resume',type=Path)
    for name in ('context','width','heads','layers','feed-forward','vocab-size'):
        train.add_argument('--'+name,type=positive)
    train.add_argument('--seed',type=int)
    train.add_argument('--learning-rate',type=float)
    train.add_argument('--tokenizer',choices=['byte','bpe'])
    evaluate=sub.add_parser('evaluate',help='Evaluate fixed weights on a supplied document split')
    evaluate.add_argument('checkpoint',type=Path)
    evaluate.add_argument('--data',type=Path,default=ROOT/'data/validation.txt')
    generation=sub.add_parser('generate',help='Generate a bounded continuation with the saved tokenizer')
    generation.add_argument('checkpoint',type=Path)
    generation.add_argument('--prompt',required=True)
    generation.add_argument('--max-new-tokens',type=positive,default=16)
    generation.add_argument('--temperature',type=float,default=0.0)
    generation.add_argument('--seed',type=int,default=23)
    return parser


def main():
    parser=build_parser(); args=parser.parse_args()
    try:
        if args.command=='check':
            from .numerics import run_checks
            output_json(run_checks())
        elif args.command=='tokenize':
            tokenizer=tokenizer_for(args.tokenizer,args.train_file,args.vocab_size)
            ids=tokenizer.encode(args.text)
            output_json({'kind':tokenizer.kind,'vocabulary':tokenizer.vocab_size,'token_ids':ids,
                         'tokens':len(ids),'utf8_bytes':len(args.text.encode('utf-8')),
                         'decoded':tokenizer.decode(ids,errors='strict'),'tokenizer_sha256':tokenizer.digest()})
        elif args.command=='train': command_train(args)
        else:
            model,_,_,metadata=load_checkpoint(args.checkpoint); tokenizer=checkpoint_tokenizer(metadata)
            if args.command=='generate':
                output_json(generate(model,args.prompt,args.max_new_tokens,args.temperature,args.seed,tokenizer))
            else:
                docs=read_documents(args.data,model.config.context,tokenizer); loss=evaluation_loss(model,docs)
                output_json({'result_class':'fixed-checkpoint-evaluation','nll_nats_per_token':loss,
                             'perplexity':math.exp(loss),'documents':len(docs),'data_sha256':digest(args.data),
                             'tokenizer_kind':tokenizer.kind,'checkpoint_step':metadata['step'],
                             'evaluation':'8 deterministic batches, batch size4, seed2026; not full-corpus perplexity'})
    except (OSError,ValueError,TypeError,KeyError,FloatingPointError) as error:
        parser.exit(2,f'Error: {error}\n')
    return 0


if __name__=='__main__': raise SystemExit(main())
