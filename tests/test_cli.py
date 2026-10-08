"""Command-level contracts that are not covered by numerical unit tests."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def command(self,*args):
        return subprocess.run([sys.executable,'-E','-B','-m','llm_lab',*args],cwd=ROOT,
                              capture_output=True,text=True,timeout=30)

    def test_tokenization_command_handles_unseen_unicode(self):
        for kind in ('byte','bpe'):
            result=self.command('tokenize','--text','A café and 雨','--tokenizer',kind,'--vocab-size','280')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['decoded'],'A café and 雨')

    def test_cli_resume_matches_one_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as directory:
            first,second,whole=[str(Path(directory)/name) for name in ('first','second','whole')]
            for arguments in [('train','--steps','3','--output',first),
                              ('train','--resume',first,'--steps','4','--output',second),
                              ('train','--steps','7','--output',whole)]:
                result=self.command(*arguments); self.assertEqual(result.returncode,0,result.stderr)
            with np.load(Path(second)/'checkpoint.npz') as resumed,np.load(Path(whole)/'checkpoint.npz') as original:
                for key in original.files: np.testing.assert_array_equal(original[key],resumed[key])
            meta=json.loads((Path(second)/'checkpoint.json').read_text())
            self.assertEqual(meta['step'],7)
            self.assertEqual(meta['tokenizer']['kind'],'byte')

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/'keep.txt'; marker.write_text('original')
            result=self.command('train','--steps','1','--output',directory)
            self.assertEqual(result.returncode,2)
            self.assertEqual(marker.read_text(),'original')
            self.assertEqual(len(list(Path(directory).iterdir())),1)

    def test_resume_rejects_changed_data_and_ignored_options(self):
        with tempfile.TemporaryDirectory() as directory:
            original=str(Path(directory)/'first'); output=str(Path(directory)/'second')
            result=self.command('train','--steps','1','--output',original)
            self.assertEqual(result.returncode,0,result.stderr)
            for args in [('--width','32'),('--train-file',str(ROOT/'data/validation.txt'))]:
                rejected=self.command('train','--resume',original,'--steps','1','--output',output,*args)
                self.assertEqual(rejected.returncode,2,rejected.stdout)
                self.assertFalse(Path(output).exists())

    def test_byte_bpe_checkpoint_metadata_and_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            result=self.command('train','--steps','2','--tokenizer','bpe','--vocab-size','280','--output',directory)
            self.assertEqual(result.returncode,0,result.stderr)
            metadata=json.loads((Path(directory)/'checkpoint.json').read_text())
            self.assertEqual(metadata['config']['vocabulary'],280)
            result=self.command('generate',directory,'--prompt','The','--max-new-tokens','3')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['output_tokens'],3)
            metadata['tokenizer']['merges'][0][0]=0
            (Path(directory)/'checkpoint.json').write_text(json.dumps(metadata))
            rejected=self.command('generate',directory,'--prompt','The')
            self.assertEqual(rejected.returncode,2)


if __name__=='__main__': unittest.main()
