"""A complete configurable-depth CPU decoder with explicit NumPy backpropagation.

This educational byte model uses learned positions, pre-RMS normalization,
multi-head causal attention, SwiGLU, and an untied vocabulary head. It is not a
pretrained assistant. All parameters are trained; no automatic differentiation
library, network service, or downloaded weights are required.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .numerics import softmax


@dataclass(frozen=True)
class Config:
    vocabulary: int = 256
    layers: int = 2
    context: int = 32
    width: int = 16
    heads: int = 2
    feed_forward: int = 32
    epsilon: float = 1e-6

    def __post_init__(self):
        for key,value in asdict(self).items():
            if key!='epsilon' and (type(value) is not int or value<1):
                raise ValueError(f'{key} must be a positive integer')
        if self.width % self.heads or not np.isfinite(self.epsilon) or self.epsilon<=0:
            raise ValueError('heads must divide width; epsilon must be positive')


def sigmoid(x):
    result=np.empty_like(x)
    positive=x>=0
    result[positive]=1/(1+np.exp(-x[positive]))
    z=np.exp(x[~positive]); result[~positive]=z/(1+z)
    return result


def norm_forward(x,g,epsilon):
    inverse=1/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+epsilon)
    return x*inverse*g,(x,g,inverse)


def norm_backward(dy,tape):
    x,g,inverse=tape
    dg=np.sum(dy*x*inverse,axis=(0,1))
    scaled=dy*g
    dx=scaled*inverse-x*inverse**3*np.mean(scaled*x,axis=-1,keepdims=True)
    return dx,dg


class TinyDecoder:
    def __init__(self,config=None,seed=17):
        self.config=config or Config()
        c=self.config; rng=np.random.default_rng(seed)
        def weight(shape): return rng.normal(0,.08,size=shape)
        self.parameters={
            'embedding':weight((c.vocabulary,c.width)), 'position':weight((c.context,c.width)),
            'g3':np.ones(c.width),'head':weight((c.width,c.vocabulary))}
        for layer in range(c.layers):
            block={'g1':np.ones(c.width), 'g2':np.ones(c.width)}
            block.update({name:weight((c.width,c.width)) for name in ('wq','wk','wv','wo')})
            block.update(wu=weight((c.width,c.feed_forward)),wg=weight((c.width,c.feed_forward)),
                         wd=weight((c.feed_forward,c.width)))
            self.parameters.update({f'block_{layer}_{name}':value for name,value in block.items()})

    def block_parameters(self,layer):
        prefix=f'block_{layer}_'
        return {name[len(prefix):]:value for name,value in self.parameters.items() if name.startswith(prefix)}

    def block_forward(self,x,p,positions,previous):
        c=self.config; batch,time,width=x.shape
        n1,t1=norm_forward(x,p['g1'],c.epsilon)
        def heads(a): return a.reshape(batch,time,c.heads,c.width//c.heads).transpose(0,2,1,3)
        q,k,v=[heads(n1@p[name]) for name in ('wq','wk','wv')]
        if previous is not None:
            k=np.concatenate((previous['k'],k),axis=2); v=np.concatenate((previous['v'],v),axis=2)
        allowed=np.arange(k.shape[2])[None,:]<=positions[:,None]
        scores=q@k.swapaxes(-1,-2)/np.sqrt(c.width//c.heads)
        a=softmax(np.where(allowed,scores,-np.inf))
        z=(a@v).transpose(0,2,1,3).reshape(batch,time,c.width)
        h=x+z@p['wo']
        n2,t2=norm_forward(h,p['g2'],c.epsilon)
        up,gate=n2@p['wu'],n2@p['wg']
        sig=sigmoid(gate); activated=gate*sig; product=activated*up
        h2=h+product@p['wd']
        tape=(n1,t1,q,k,v,a,z,n2,t2,up,gate,sig,activated,product)
        return h2,{'k':k.copy(),'v':v.copy()},tape

    def block_backward(self,dh2,tape,p):
        c=self.config
        n1,t1,q,k,v,a,z,n2,t2,up,gate,sig,activated,product=tape
        gradients={}
        def linear(dy,inputs,name):
            gradients[name]=inputs.reshape(-1,inputs.shape[-1]).T@dy.reshape(-1,dy.shape[-1])
            return dy@p[name].T
        dproduct=linear(dh2,product,'wd')
        dactivated=dproduct*up; dup=dproduct*activated
        dgate=dactivated*(sig+gate*sig*(1-sig))
        dn2=linear(dup,n2,'wu')+linear(dgate,n2,'wg')
        dhnorm,gradients['g2']=norm_backward(dn2,t2)
        dh=dh2+dhnorm
        dz=linear(dh,z,'wo')
        batch,time,_=dh.shape; head_width=c.width//c.heads
        dz=dz.reshape(batch,time,c.heads,head_width).transpose(0,2,1,3)
        da=dz@v.swapaxes(-1,-2); dv=a.swapaxes(-1,-2)@dz
        ds=a*(da-np.sum(da*a,axis=-1,keepdims=True))
        dq=ds@k/np.sqrt(head_width); dk=ds.swapaxes(-1,-2)@q/np.sqrt(head_width)
        def joined(t): return t.transpose(0,2,1,3).reshape(batch,time,c.width)
        dn1=linear(joined(dq),n1,'wq')+linear(joined(dk),n1,'wk')+linear(joined(dv),n1,'wv')
        dxnorm,gradients['g1']=norm_backward(dn1,t1)
        return dh+dxnorm,gradients

    def forward(self,tokens,cache=None,retain=False):
        tokens=np.asarray(tokens)
        c,p=self.config,self.parameters
        if tokens.ndim!=2 or tokens.size==0 or not np.issubdtype(tokens.dtype,np.integer):
            raise ValueError('tokens must be a nonempty integer [batch,time] array')
        if (tokens<0).any() or (tokens>=c.vocabulary).any(): raise ValueError('token outside vocabulary')
        batch,time=tokens.shape; offset=0; prior=[None]*c.layers
        if cache is not None:
            if retain: raise ValueError('training does not support a supplied cache')
            if not isinstance(cache,dict) or set(cache)!={'layers','length'}:
                raise ValueError('invalid cache structure')
            offset=cache['length']; prior=cache['layers']
            if type(offset) is not int or offset<0 or len(prior)!=c.layers: raise ValueError('invalid cache length')
            expected=(batch,c.heads,offset,c.width//c.heads)
            for layer in prior:
                if set(layer)!={'k','v'} or any(a.shape!=expected or not np.isfinite(a).all() for a in layer.values()):
                    raise ValueError('cache tensor axes or values do not match')
        if offset+time>c.context: raise ValueError('context limit exceeded; begin a new explicit context')
        positions=np.arange(offset,offset+time)
        x=p['embedding'][tokens]+p['position'][positions]
        tapes=[]; layers=[]
        for layer in range(c.layers):
            x,state,tape=self.block_forward(x,self.block_parameters(layer),positions,prior[layer])
            tapes.append(tape); layers.append(state)
        n3,t3=norm_forward(x,p['g3'],c.epsilon)
        logits=n3@p['head']
        if retain: return logits,(tokens,positions,tapes,n3,t3)
        return logits,{'layers':layers,'length':offset+time}

    def loss_grad(self,tokens,targets):
        logits,tape=self.forward(tokens,retain=True)
        targets=np.asarray(targets)
        if targets.shape!=logits.shape[:-1] or not np.issubdtype(targets.dtype,np.integer):
            raise ValueError('targets must be integer [batch,time]')
        if (targets<0).any() or (targets>=self.config.vocabulary).any(): raise ValueError('invalid target')
        shifted=logits-logits.max(axis=-1,keepdims=True)
        probs=softmax(logits)
        loss=float(np.mean(np.log(np.exp(shifted).sum(axis=-1))-np.take_along_axis(shifted,targets[...,None],axis=-1)[...,0]))
        dlogits=probs.copy()
        np.put_along_axis(dlogits,targets[...,None],np.take_along_axis(dlogits,targets[...,None],axis=-1)-1,axis=-1)
        dlogits/=targets.size
        tokens,positions,tapes,n3,t3=tape; p=self.parameters
        gradients={'head':n3.reshape(-1,n3.shape[-1]).T@dlogits.reshape(-1,dlogits.shape[-1])}
        dx,gradients['g3']=norm_backward(dlogits@p['head'].T,t3)
        for layer in reversed(range(self.config.layers)):
            dx,block=self.block_backward(dx,tapes[layer],self.block_parameters(layer))
            gradients.update({f'block_{layer}_{name}':value for name,value in block.items()})
        gradients['embedding']=np.zeros_like(p['embedding']); np.add.at(gradients['embedding'],tokens,dx)
        gradients['position']=np.zeros_like(p['position']); gradients['position'][positions]=dx.sum(axis=0)
        if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients.values()):
            raise FloatingPointError('nonfinite loss or gradient')
        return loss,gradients


class Adam:
    """Adam with bias correction and optional global norm clipping; no decay."""
    def __init__(self,parameters,learning_rate=.003):
        if not np.isfinite(learning_rate) or learning_rate<=0: raise ValueError('positive learning rate required')
        self.learning_rate=learning_rate; self.step=0
        self.m={k:np.zeros_like(v) for k,v in parameters.items()}
        self.v={k:np.zeros_like(v) for k,v in parameters.items()}

    def update(self,parameters,gradients,clip=1.0):
        if set(parameters)!=set(gradients) or not np.isfinite(clip) or clip<=0: raise ValueError('invalid gradients or clip')
        for k in parameters:
            if gradients[k].shape!=parameters[k].shape or not np.isfinite(gradients[k]).all(): raise ValueError('invalid gradient array')
        maximum=max(float(np.max(np.abs(g))) for g in gradients.values())
        scaled_norm=float(np.sqrt(sum(float(np.sum((g/maximum)**2)) for g in gradients.values()))) if maximum else 0.0
        norm=maximum*scaled_norm
        if not np.isfinite(norm): raise FloatingPointError('gradient norm exceeds the representable range')
        factor=min(1.,(clip/maximum)/scaled_norm) if maximum else 1.0
        self.step+=1
        for k,p in parameters.items():
            g=gradients[k]*factor
            self.m[k]=.9*self.m[k]+.1*g; self.v[k]=.999*self.v[k]+.001*g*g
            corrected_m=self.m[k]/(1-.9**self.step); corrected_v=self.v[k]/(1-.999**self.step)
            p-=self.learning_rate*corrected_m/(np.sqrt(corrected_v)+1e-8)
        return norm


def read_documents(path,context,tokenizer=None):
    if type(context) is not int or context < 1: raise ValueError('context must be a positive integer')
    if tokenizer is None:
        from .tokenization import ByteTokenizer
        tokenizer=ByteTokenizer()
    docs=[np.asarray(tokenizer.encode(text),dtype=np.int64)
          for text in Path(path).read_text(encoding='utf-8').split('\n\n') if text.strip()]
    if not docs or any(len(doc)<context+1 for doc in docs):
        raise ValueError(f'each document needs at least {context+1} tokens')
    return docs


def sample_batch(documents,rng,context,batch=4):
    if type(context) is not int or context<1 or type(batch) is not int or batch<1:
        raise ValueError('context and batch must be positive integers')
    if not documents or any(not isinstance(doc,np.ndarray) or doc.ndim!=1 or not np.issubdtype(doc.dtype,np.integer) or len(doc)<context+1 for doc in documents):
        raise ValueError('documents must be integer vectors longer than the context')
    blocks=[]
    for _ in range(batch):
        doc=documents[int(rng.integers(len(documents)))]; start=int(rng.integers(len(doc)-context))
        blocks.append(doc[start:start+context+1])
    block=np.stack(blocks)
    return block[:,:-1],block[:,1:]


def train_steps(model,optimizer,documents,rng,steps):
    if type(steps) is not int or steps<1: raise ValueError('steps must be positive integer')
    history=[]
    for _ in range(steps):
        x,y=sample_batch(documents,rng,model.config.context)
        loss,gradients=model.loss_grad(x,y)
        norm=optimizer.update(model.parameters,gradients)
        history.append({'step':optimizer.step,'loss':loss,'gradient_norm_before_clip':norm})
    return history


def evaluation_loss(model,documents):
    rng=np.random.default_rng(2026)
    values=[]
    for _ in range(8):
        x,y=sample_batch(documents,rng,model.config.context)
        logits,_=model.forward(x)
        shifted=logits-logits.max(axis=-1,keepdims=True)
        values.append(float(np.mean(np.log(np.exp(shifted).sum(axis=-1))-np.take_along_axis(shifted,y[...,None],axis=-1)[...,0])))
    return float(np.mean(values))


def save_checkpoint(directory,model,optimizer,rng,data_digests,tokenizer=None):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    if any((directory/name).exists() for name in ('checkpoint.npz','checkpoint.json')):
        raise FileExistsError('checkpoint destination already exists')
    arrays={prefix+k:v for prefix,group in [('p_',model.parameters),('m_',optimizer.m),('v_',optimizer.v)] for k,v in group.items()}
    target=directory/'checkpoint.npz'; np.savez_compressed(target,**arrays)
    metadata={'format_version':1,'config':asdict(model.config),'step':optimizer.step,
              'learning_rate':optimizer.learning_rate,'rng_state':rng.bit_generator.state,
              'tokenizer':tokenizer.to_dict() if tokenizer is not None else None,
              'tokenizer_sha256':tokenizer.digest() if tokenizer is not None else None,
              'data_digests':data_digests,
              'numpy':np.__version__,'weights_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (directory/'checkpoint.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load_checkpoint(directory):
    directory=Path(directory); metadata=json.loads((directory/'checkpoint.json').read_text())
    if metadata.get('format_version')!=1: raise ValueError('unsupported checkpoint format')
    if metadata.get('tokenizer') is not None:
        from .tokenization import tokenizer_from_dict
        tokenizer=tokenizer_from_dict(metadata['tokenizer'])
        if tokenizer.digest()!=metadata.get('tokenizer_sha256') or tokenizer.vocab_size!=metadata['config']['vocabulary']:
            raise ValueError('tokenizer identity does not match checkpoint')
    data=(directory/'checkpoint.npz').read_bytes()
    if hashlib.sha256(data).hexdigest()!=metadata['weights_sha256']: raise ValueError('checkpoint digest mismatch')
    model=TinyDecoder(Config(**metadata['config'])); optimizer=Adam(model.parameters,metadata['learning_rate'])
    optimizer.step=metadata['step']
    if type(optimizer.step) is not int or optimizer.step<0: raise ValueError('invalid optimizer step')
    with np.load(directory/'checkpoint.npz',allow_pickle=False) as archive:
        expected={prefix+k for prefix in ('p_','m_','v_') for k in model.parameters}
        if set(archive.files)!=expected: raise ValueError('checkpoint tensor set mismatch')
        for prefix,group in [('p_',model.parameters),('m_',optimizer.m),('v_',optimizer.v)]:
            for k in group:
                value=archive[prefix+k]
                if value.shape!=group[k].shape or value.dtype!=np.float64 or not np.isfinite(value).all(): raise ValueError('invalid checkpoint tensor')
                if prefix=='v_' and (value<0).any(): raise ValueError('negative Adam second moment')
                group[k]=value.copy()
    rng=np.random.default_rng(); rng.bit_generator.state=metadata['rng_state']
    return model,optimizer,rng,metadata


def generate(model,prompt,max_new_tokens=16,temperature=0.0,seed=23,tokenizer=None):
    if tokenizer is None:
        from .tokenization import ByteTokenizer
        tokenizer=ByteTokenizer()
    if model.config.vocabulary!=tokenizer.vocab_size: raise ValueError('tokenizer vocabulary does not match model')
    encoded=tokenizer.encode(prompt)
    if not encoded: raise ValueError('prompt must contain at least one byte')
    if type(max_new_tokens) is not int or max_new_tokens<1 or len(encoded)+max_new_tokens>model.config.context:
        raise ValueError('prompt tokens plus requested tokens must fit the context')
    if not np.isfinite(temperature) or temperature<0: raise ValueError('temperature must be nonnegative')
    rng=np.random.default_rng(seed); logits,cache=model.forward(np.array([encoded]))
    generated=[]
    for i in range(max_new_tokens):
        last=logits[0,-1]
        if temperature == 0:
            token = int(last.argmax())
        else:
            # Center first: a tiny positive temperature must not create +inf.
            with np.errstate(over='ignore', under='ignore'):
                distribution = softmax((last - last.max()) / temperature)
            token = int(rng.choice(model.config.vocabulary, p=distribution))
        generated.append(token)
        if i+1<max_new_tokens: logits,cache=model.forward(np.array([[token]]),cache)
    return {'text':tokenizer.decode(encoded+generated),'generated_token_ids':generated,
            'prompt_tokens':len(encoded),'output_tokens':len(generated),'tokenizer':tokenizer.kind,
            'interpretation':'Small-model continuation; not evidence of factual correctness or instruction-following capability.'}
