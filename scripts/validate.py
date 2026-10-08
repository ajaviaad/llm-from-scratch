#!/usr/bin/env python3
"""Run unit tests and both real tokenizer/training paths without network access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def validate(output):
    import numpy as np
    record={'status':'running','python':platform.python_version(),'numpy':np.__version__,
            'platform':platform.platform(),'steps':[],
            'scope':'CPU correctness, tiny-corpus training, checkpoint and command integration; no large-model capability claim.'}
    def command(name,args):
        process=subprocess.run([sys.executable,'-E','-B','-X','utf8',*args],cwd=ROOT,
                               capture_output=True,text=True,encoding='utf-8',timeout=180)
        (output/(name+'.stdout.txt')).write_text(process.stdout,encoding='utf-8')
        (output/(name+'.stderr.txt')).write_text(process.stderr,encoding='utf-8')
        record['steps'].append({'name':name,'exit_code':process.returncode})
        if process.returncode:
            raise RuntimeError(name+': '+process.stderr[-4000:])
        return process
    try:
        print('Checking numerical, tokenizer, model, and training invariants ...',flush=True)
        tests=command('unit_tests',['-m','unittest','discover','-s','tests','-v'])
        count=re.search(r'Ran (\d+) tests? in ',tests.stderr)
        if not count or int(count.group(1))<1: raise RuntimeError('No unit tests were confirmed')
        record['unit_tests']=int(count.group(1))
        arithmetic=json.loads(command('numerics',['-m','llm_lab','check']).stdout)
        if not arithmetic['passed']: raise RuntimeError('Numerical checks failed')
        record['numerical_checks']=arithmetic['total']
        for kind in ('byte','bpe'):
            print(f'Training and exercising the {kind} model ...',flush=True)
            location=output/kind
            args=['-m','llm_lab','train','--steps','120','--tokenizer',kind,'--output',str(location)]
            if kind=='bpe': args+=['--vocab-size','300']
            command(kind+'_training',args)
            result=json.loads((location/'training.json').read_text())
            if not result['final_validation_nll']<result['initial_validation_nll']:
                raise RuntimeError(kind+' instructional run did not improve validation loss')
            sample=json.loads(command(kind+'_generation',['-m','llm_lab','generate',str(location),'--prompt','The policy','--max-new-tokens','8']).stdout)
            if sample['output_tokens']!=8: raise RuntimeError('Unexpected generation length')
            evaluation=json.loads(command(kind+'_evaluation',['-m','llm_lab','evaluate',str(location)]).stdout)
            if abs(evaluation['nll_nats_per_token']-result['final_validation_nll'])>1e-12:
                raise RuntimeError('Saved checkpoint evaluation disagrees with training report')
            command(kind+'_resume',['-m','llm_lab','train','--resume',str(location),'--steps','5','--output',str(output/(kind+'-resumed'))])
            record[kind]={'parameters':result['parameter_count'],'initial_validation_nll':result['initial_validation_nll'],
                          'final_validation_nll':result['final_validation_nll'],'generation_tokens':8,'resume_step':125}
        record['source_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted((ROOT/'llm_lab').glob('*.py'))}
        record['status']='passed'; print(f"PASS: {record['unit_tests']} tests; {record['numerical_checks']} reference checks; byte and BPE training, evaluation, generation and resume.")
        return 0
    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as error:
        record['status']='failed'; record['error']=str(error); print(str(error),file=sys.stderr); return 1
    finally:
        write_json(output/'validation.json',record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='retain evidence in a new/empty directory (otherwise temporary)')
    args=parser.parse_args()
    if sys.version_info<(3,11): parser.error('Python 3.11 or newer required')
    try:
        import numpy
    except ImportError:
        parser.error('NumPy is missing; install requirements.txt using this interpreter')
    if args.output:
        path=args.output.expanduser().resolve()
        if path.exists() and (not path.is_dir() or any(path.iterdir())): parser.error('output must be new or empty')
        path.mkdir(parents=True,exist_ok=True)
        return validate(path)
    with tempfile.TemporaryDirectory(prefix='llm-scratch-validation-') as directory:
        status=validate(Path(directory))
    print('Temporary evidence removed. Use --output to retain it.')
    return status


if __name__=='__main__': raise SystemExit(main())
