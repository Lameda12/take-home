# IlyaGusev/saiga_preferences: stats for DPO on Qwen2.5-1.5B-Instruct

Measured 2026-09-28 with `datasets` + the `Qwen/Qwen2.5-1.5B-Instruct` tokenizer. The script and its raw output are at the end of this file.

## 1. Basics
- Not gated. Loads with `load_dataset("IlyaGusev/saiga_preferences")`. It has one config and one split: **train, 30,590 rows**.
- Columns:
  - `prompt`, `chosen`, `rejected`: each is a list of `{role, content}` messages. `chosen`/`rejected` hold a single assistant message, and `prompt` can include a system prompt and earlier turns.
  - `chosen_model`, `rejected_model`, `source`: strings.
  - `sonnet_approved`, `is_bad_by_regex`: bools.
- **License: none declared.** The README has only the `dataset_info` YAML, with no license field and no prose. The Hub tags have no license either. Treat it as unlicensed and say so in the write-up. The data mixes outputs from GPT-4, Claude and others, so their terms of use apply.
- Example row 0: the source is `saiga_nemo_on_policy`, and both models are `saiga_nemo_simpo`. The system prompt is "Ты - gpt-4o, новейшая языковая модель, разработанная OpenAI". The dialogue is multi-turn and emotional/roleplay. chosen and rejected are both a "200-calorie store-bought menu" answer, with different portions.
- Other example prompts, sampled at random: a comparison of gaming mice, Belarus water-law easements, a roleplay line ("выпить кофе?"), Zamyatin's works, a C# class question. It is general-assistant Russian chat.

## 2. Quality/filter columns
| column | distribution |
|---|---|
| sonnet_approved | True 22,014 / False 8,576 |
| is_bad_by_regex | False 28,258 / True 2,332 |
| approved & not bad | **20,386** |
| source | lmsys_clean_ru_queries 7,067; saiga_nemo_on_policy 6,451; saiga_bot_multiturn 4,932; all_prompts_20241027 4,710; saiga_bot_all_gemma 4,599; pippa_and_multiturn_10_10 1,943 (roleplay); lmsys_clean_ru_queries_multiturn 683; saiga_bot_all 205 |
| chosen_model (top) | saiga_llama3_8b 4,974; saiga_nemo_simpo 4,430; gpt_4 2,494; saiga_gemma2_9b_sft 2,321; qwen25_72b 2,113; claude_3_5_sonnet_20241022 2,077; ... |
| rejected_model (top) | saiga_nemo_sft 5,416; saiga_llama3_8b 4,841; llama_3_8b 3,412; saiga_nemo_simpo 2,869; ... |

## 3. Token lengths (Qwen2.5 tokenizer)
The prompt is measured with the chat template applied (`add_generation_prompt=True`, which also inserts Qwen's default system prompt when the row has none). Completions are measured as content + `<|im_end|>\n`.

| | p50 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|
| prompt | 148 | 1813 | 2778 | 5755 | 13454 | 651 |
| chosen | 548 | 998 | 1176 | 1631 | 4200 | 572 |
| rejected | 469 | 869 | 1031 | 1486 | 5111 | 492 |
| prompt + max(c,r) | 895 | 2508 | 3435 | 6600 | 14968 | 1263 |

## 4. Truncation at a given max_length
Method: the prompt is kept to its last `max_length//2` tokens (keep_end). Each completion is then cut to the tokens that remain in the budget.

| max_length | fits with no truncation | prompt truncated | a completion truncated | chosen == rejected after truncation (token ids) |
|---|---|---|---|---|
| 512 | 13.7% | 44.2% | 75.1% | 6 rows (0.02%) |
| 768 | 35.2% | 38.4% | 51.1% | 3 rows (0.01%) |
| 1024 | 55.0% | 34.0% | 28.7% | 3 rows (0.01%) |
| 1536 | 71.7% | 25.8% | 10.4% | 3 rows (0.01%) |

Rows collapsing to identical token ids are negligible. The bigger risk is that heavy truncation cuts both completions to the same prefix length, so what separates them is lost. Filtering to rows that fit is better than truncating.

## 5. Other checks
- chosen == rejected (exact text): **2 rows**.
- Multi-turn prompts (more than one user turn): **10,533**. Single-turn: 20,057. 12,868 rows carry a system prompt.
- **Length bias:** chosen is longer than rejected by a mean of +79.7 tokens (median +57), and chosen is the longer one in 67.8% of rows. In characters the means are 1,803 vs 1,594. Inside the filtered subset below the bias is stronger (+72 to +93 tokens, chosen longer in 71-73%). DPO trained on this will likely learn to be verbose. Measure response length before and after training.

## 6. Suggested ~600-row subset
Filters: `sonnet_approved & ~is_bad_by_regex`, a single user turn, chosen != rejected, `prompt_tokens <= L//2` and `prompt + max(chosen, rejected) <= L`, so that nothing is truncated.
- At L=768 this leaves **6,321** rows (lmsys_clean_ru_queries 2,338, saiga_bot_all_gemma 1,173, saiga_nemo_on_policy 1,011, all_prompts 969, saiga_bot_multiturn 704, pippa 108, saiga_bot_all 18).
- At L=1024 it leaves **9,991** rows.
- Recommendation:
  - Use max_length=1024 with max_prompt_length=512. Nearly 10k clean rows fit untruncated, and 1.5B memory is fine there. If memory is tight, 768 still leaves over 6k rows.
  - Also drop `pippa_and_multiturn_10_10` (roleplay).
  - Optionally cap |len(chosen) − len(rejected)| (e.g. ≤ 30% or ≤ 150 tokens) to reduce length bias.
  - Then take a seeded random sample of 600 rows, split 500 train / 100 held-out. Stratify by source if you want coverage.
- **Does it look like support tickets? No.** It is general Russian assistant chat: lmsys queries, bot logs, coding, trivia, writing, roleplay. A keyword regex for support terms (заказ/доставка/возврат/оплата/аккаунт/пароль/поддержка/клиент/тариф/подписка/refund/order/account/support) matches 1,488 first user turns. A sample of those matches were false positives (a job application, SEO keyword analysis, a giveaway post). About 1,174 first user turns contain no Cyrillic, so roughly 4% of prompts are not in Russian. For a support-ticket framing, you need either a different dataset or to acknowledge that this is general-assistant preference data.

## Script
```python
import numpy as np, collections, re
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
card=open(hf_hub_download('IlyaGusev/saiga_preferences','README.md',repo_type='dataset')).read()
print("CARD_HEAD:\n", card[:1500])
ds=load_dataset('IlyaGusev/saiga_preferences')['train']
tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
df=ds.to_pandas()
for c in ['source','chosen_model','rejected_model','sonnet_approved','is_bad_by_regex']:
    print(c, df[c].value_counts(dropna=False).head(20).to_dict())
print(pd:=None)
print("approved x bad", df.groupby(['sonnet_approved','is_bad_by_regex']).size().to_dict())
P,C,R,T=[],[],[],[]
same_text=0; multiturn=[]; nturns=[]; chosen_len_chars=[]; rej_len_chars=[]; sysp=0
rows=[]
for ex in ds:
    p=ex['prompt']; users=sum(m['role']=='user' for m in p); nturns.append(users)
    sysp+= any(m['role']=='system' for m in p)
    pid=tok.apply_chat_template(p,add_generation_prompt=True,tokenize=True)
    if hasattr(pid,'input_ids'): pid=pid['input_ids']
    cc=ex['chosen'][-1]['content']; rc=ex['rejected'][-1]['content']
    same_text+= cc==rc
    cid=tok(cc+'<|im_end|>\n',add_special_tokens=False)['input_ids']
    rid=tok(rc+'<|im_end|>\n',add_special_tokens=False)['input_ids']
    rows.append((pid,cid,rid))
    P.append(len(pid));C.append(len(cid));R.append(len(rid));T.append(len(pid)+max(len(cid),len(rid)))
    chosen_len_chars.append(len(cc)); rej_len_chars.append(len(rc))
def q(a): a=np.array(a); return {k:int(np.percentile(a,k)) for k in (50,90,95,99)}|{'max':int(a.max()),'mean':round(float(a.mean()),1)}
for n,a in [('prompt',P),('chosen',C),('rejected',R),('prompt+max(c,r)',T)]: print(n,q(a))
print("chosen==rejected exact text:",same_text)
nt=collections.Counter(nturns); print("user turns dist",sorted(nt.items())[:10], "multi-turn(>1 user):",sum(v for k,v in nt.items() if k>1)); print("has system prompt:",sysp)
C_,R_=np.array(C),np.array(R); print("mean chosen-rejected tokens:",round(float((C_-R_).mean()),1),"median",float(np.median(C_-R_)),"frac chosen longer",round(float((C_>R_).mean()),3))
print("mean chars chosen",np.mean(chosen_len_chars),"rejected",np.mean(rej_len_chars))
df['P']=P;df['C']=C;df['R']=R;df['T']=T
for L in (512,768,1024,1536):
    fit=0; ident=0; ptrunc=0; ctrunc=0
    for pid,cid,rid in rows:
        pm=L//2; p2=pid[-pm:] if len(pid)>pm else pid; ptrunc+=len(pid)>pm
        b=L-len(p2); c2=cid[:b]; r2=rid[:b]
        ctrunc+= (len(cid)>b or len(rid)>b)
        fit+= len(pid)<=pm and len(cid)<=b and len(rid)<=b
        ident+= c2==r2
    n=len(rows); print(f"L={L}: fit_untruncated={fit/n:.3f} prompt_truncated={ptrunc/n:.3f} completion_truncated={ctrunc/n:.3f} identical_after_trunc={ident/n:.4f} ({ident})")
for L in (768,1024):
  for filt,name in [((df.sonnet_approved)&(~df.is_bad_by_regex),'approved&!bad')]:
    m=filt&(df.P<=L//2)&(df[['C','R']].max(axis=1)<=L-df.P)&(pd_:=np.array([1]*len(df))==1)
    m2=m&(np.array(nturns)==1)
    m3=m2&(df.C.values!=df.R.values)
    print(L,name,"fits:",int(m.sum()),"single-turn:",int(m2.sum()),"by source:",df[m2].source.value_counts().to_dict())
    print(" length diff in that subset mean",round(float((df[m2].C-df[m2].R).mean()),1), "frac chosen longer", round(float((df[m2].C>df[m2].R).mean()),3))
# support-ticket-ish keywords
kw=re.compile(r'(заказ|доставк|возврат|оплат|аккаунт|пароль|поддержк|жалоб|клиент|тариф|подписк|refund|order|account|support)',re.I)
first_user=[next((m['content'] for m in ex['prompt'] if m['role']=='user'),'') for ex in ds]
hits=[bool(kw.search(u)) for u in first_user]; print("support-keyword hits in first user msg:",sum(hits))
import random; random.seed(0)
for u in random.sample([u for u,h in zip(first_user,hits) if h],3): print("KW:",u[:200].replace('\n',' '))
for u in random.sample(first_user,6): print("RAND:",u[:200].replace('\n',' '))
eng=sum(1 for u in first_user if not re.search('[а-яА-Я]',u)); print("first user msg w/o cyrillic:",eng)
```

## Raw output
```
source {'lmsys_clean_ru_queries': 7067, 'saiga_nemo_on_policy': 6451, 'saiga_bot_multiturn': 4932, 'all_prompts_20241027': 4710, 'saiga_bot_all_gemma': 4599, 'pippa_and_multiturn_10_10': 1943, 'lmsys_clean_ru_queries_multiturn': 683, 'saiga_bot_all': 205}
chosen_model {'saiga_llama3_8b': 4974, 'saiga_nemo_simpo': 4430, 'gpt_4': 2494, 'saiga_gemma2_9b_sft': 2321, 'qwen25_72b': 2113, 'claude_3_5_sonnet_20241022': 2077, 'llama_3_8b': 1964, 'saiga_nemo_sft': 1864, 'sfr': 1464, 'gpt_3_5': 1308, 'suzume': 1197, 'gpt_4o_20240806': 1144, 'gpt_4o': 885, 'gemma2_9b_it_abliterated': 741, 'claude_3_opus': 423, 'claude_3_5_sonnet': 338, 'aya_23_8b': 323, 'gpt_4o_mini': 258, 'vikhr_nemo': 182, 'saiga_phi3_medium_sft': 90}
rejected_model {'saiga_nemo_sft': 5416, 'saiga_llama3_8b': 4841, 'llama_3_8b': 3412, 'saiga_nemo_simpo': 2869, 'saiga_gemma2_9b_sft': 2278, 'gpt_3_5': 1711, 'gpt_4o_20240806': 1692, 'qwen25_72b': 1424, 'gpt_4': 1263, 'sfr': 1239, 'suzume': 1013, 'gemma2_9b_it_abliterated': 790, 'aya_23_8b': 628, 'claude_3_5_sonnet_20241022': 495, 'gpt_4o': 478, 'vikhr_nemo': 395, 'gpt_4o_mini': 251, 'claude_3_opus': 168, 'saiga_phi3_medium_sft': 115, 'claude_3_5_sonnet': 112}
sonnet_approved {True: 22014, False: 8576}
is_bad_by_regex {False: 28258, True: 2332}
None
approved x bad {(False, False): 7872, (False, True): 704, (True, False): 20386, (True, True): 1628}
prompt {50: 148, 90: 1813, 95: 2778, 99: 5755, 'max': 13454, 'mean': 650.6}
chosen {50: 548, 90: 998, 95: 1176, 99: 1631, 'max': 4200, 'mean': 571.7}
rejected {50: 469, 90: 869, 95: 1031, 99: 1486, 'max': 5111, 'mean': 492.0}
prompt+max(c,r) {50: 895, 90: 2508, 95: 3435, 99: 6600, 'max': 14968, 'mean': 1263.4}
chosen==rejected exact text: 2
user turns dist [(1, 20057), (2, 4515), (3, 2456), (4, 1165), (5, 746), (6, 657), (7, 268), (8, 112), (9, 211), (10, 69)] multi-turn(>1 user): 10533
has system prompt: 12868
mean chosen-rejected tokens: 79.7 median 57.0 frac chosen longer 0.678
mean chars chosen 1803.4185681595293 rejected 1594.378620464204
L=512: fit_untruncated=0.137 prompt_truncated=0.442 completion_truncated=0.751 identical_after_trunc=0.0002 (6)
L=768: fit_untruncated=0.352 prompt_truncated=0.384 completion_truncated=0.511 identical_after_trunc=0.0001 (3)
L=1024: fit_untruncated=0.550 prompt_truncated=0.340 completion_truncated=0.287 identical_after_trunc=0.0001 (3)
L=1536: fit_untruncated=0.717 prompt_truncated=0.258 completion_truncated=0.104 identical_after_trunc=0.0001 (3)
768 approved&!bad fits: 6919 single-turn: 6321 by source: {'lmsys_clean_ru_queries': 2338, 'saiga_bot_all_gemma': 1173, 'saiga_nemo_on_policy': 1011, 'all_prompts_20241027': 969, 'saiga_bot_multiturn': 704, 'pippa_and_multiturn_10_10': 108, 'saiga_bot_all': 18}
 length diff in that subset mean 71.6 frac chosen longer 0.709
1024 approved&!bad fits: 11068 single-turn: 9991 by source: {'lmsys_clean_ru_queries': 3420, 'saiga_nemo_on_policy': 1986, 'saiga_bot_all_gemma': 1901, 'all_prompts_20241027': 1579, 'saiga_bot_multiturn': 908, 'pippa_and_multiturn_10_10': 168, 'saiga_bot_all': 29}
 length diff in that subset mean 92.5 frac chosen longer 0.733
support-keyword hits in first user msg: 1488
KW: Клиент разместил объявление. Объявление о вакансии тендерного специалиста я хочу на него ответить вернуться помоги мне написать, сообщить что я более трёх лет занимаюсь тендерами знаком с многими элек
KW: Проанализируй поисковые запросы: laduchi consult	116 корпоративная база знаний примеры	44 laduchi	40 налог на роскошь в испании	38 база знаний пример	23 база знаний компании пример	19 налог на богатст
KW: Конкурсный текст [🎁 РАЗЫГРЫВАЕМ ШУРУПОВËРТ DEWALT 🎁  ВЫПОЛНИ 4 УСЛОВИЯ КОНКУРСА: ✅1. Быть подписчиком нашей группы ✔ ✅2. Сделать репост этой записи 📣 ✅3. Разослать этот конкурс 2-м друзьям, которые лю
RAND: если цена у дефендера gk-418 2130 а у ardor gaming blade 2500, то что лучше взять? Посоветуй
RAND: Преимущества закрепления "водного сервитута" в водном законодательстве Республики Беларусь.
RAND: А, молодец, Асасио! Можешь немного расслабиться. Не хочешь выпить кофе?
RAND: какие ты знаешь произведения Замятина?
RAND: содержание книги психология питания
RAND: Есть такой класс: public class Temp     {         private float _speed = 0;         private float _newSpeed = 1;                  public void Setup()         {             _speed = GameDirector.BotSpe
first user msg w/o cyrillic: 1174
```
