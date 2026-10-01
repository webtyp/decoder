"""The LFM2 decode step, token by token, in numpy: the executable spec the Go decoder follows.
Run from the decoder root: ~/Dev/LMmodels/.venv/bin/python testdata/lfm2_reference_step.py
It matches transformers on testdata/tiny_lfm2 within 2e-6."""
import json, numpy as np
from safetensors.numpy import load_file
d="testdata/tiny_lfm2/"
W=load_file(d+"model.safetensors"); cfg=json.load(open(d+"config.json")); ref=json.load(open(d+"reference.json"))
H=cfg["hidden_size"]; nh=cfg["num_attention_heads"]; kvh=cfg["num_key_value_heads"]; hd=H//nh; K=cfg["conv_L_cache"]; eps=cfg["norm_eps"]
theta=cfg["rope_parameters"]["rope_theta"]; types=cfg["layer_types"]
def rms(x,w): return w*(x/np.sqrt(np.mean(x*x)+eps))
def silu(x): return x/(1+np.exp(-x))
inv=1.0/(theta**(np.arange(0,hd,2)/hd))
def rope(v,pos):  # rotate_half over the whole head
    f=pos*inv; c=np.cos(np.concatenate([f,f])); s=np.sin(np.concatenate([f,f]))
    h=hd//2; rot=np.concatenate([-v[h:],v[:h]]); return v*c+rot*s
conv={i:np.zeros((H,K-1)) for i,t in enumerate(types) if t=="conv"}; kc={i:[] for i,t in enumerate(types) if t!="conv"}; vc={i:[] for i in kc}
maxd=0
for pos,tok in enumerate(ref["prompt"]):
    x=W["model.embed_tokens.weight"][tok].astype(np.float64)
    for i,t in enumerate(types):
        p=f"model.layers.{i}."
        h=rms(x,W[p+"operator_norm.weight"])
        if t=="conv":
            bcx=W[p+"conv.in_proj.weight"]@h; B,C,xx=bcx[:H],bcx[H:2*H],bcx[2*H:]
            bx=B*xx
            win=np.concatenate([conv[i],bx[:,None]],axis=1)  # [H,K], oldest first
            w=W[p+"conv.conv.weight"][:,0,:]                 # [H,K]
            y=(win*w).sum(1); conv[i]=win[:,1:]
            o=W[p+"conv.out_proj.weight"]@(C*y)
        else:
            q=(W[p+"self_attn.q_proj.weight"]@h).reshape(nh,hd); k=(W[p+"self_attn.k_proj.weight"]@h).reshape(kvh,hd); v=(W[p+"self_attn.v_proj.weight"]@h).reshape(kvh,hd)
            q=np.stack([rope(rms(q[j],W[p+"self_attn.q_layernorm.weight"]),pos) for j in range(nh)])
            k=np.stack([rope(rms(k[j],W[p+"self_attn.k_layernorm.weight"]),pos) for j in range(kvh)])
            kc[i].append(k); vc[i].append(v); Ks=np.stack(kc[i]); Vs=np.stack(vc[i])
            out=[]
            for j in range(nh):
                g=j//(nh//kvh); sc=Ks[:,g,:]@q[j]/np.sqrt(hd); a=np.exp(sc-sc.max()); a/=a.sum(); out.append(a@Vs[:,g,:])
            o=W[p+"self_attn.out_proj.weight"]@np.concatenate(out)
        x=x+o
        h=rms(x,W[p+"ffn_norm.weight"])
        x=x+W[p+"feed_forward.w2.weight"]@(silu(W[p+"feed_forward.w1.weight"]@h)*(W[p+"feed_forward.w3.weight"]@h))
    x=rms(x,W["model.embedding_norm.weight"])
    logits=W["model.embed_tokens.weight"]@x
    maxd=max(maxd,np.abs(logits-np.array(ref["token_by_token_logits"][pos])).max())
print("max |numpy - transformers| =", maxd)
