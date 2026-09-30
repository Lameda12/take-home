# The Take-Home, Explained Plainly

A plain-language guide to what this task is, what has been done so far, and what everything means, so you can explain it yourself in the live interview.

---

## 1. What is the task really asking?

Soup is a tool that fine-tunes AI models with one command. They ask you to:

1. Train a model using one method (**DPO**) on a cheap GPU (a **T4**) with one feature (**layer streaming**).
2. Then answer one question: **"Did this actually work, and should we ship it?"**

The trick: they warn that **a run that finishes, a loss that goes down, and Soup saying "SHIP" do not prove anything.** They want to see if you can *prove* training happened, and find the ways it could *look* fine while being broken.

So this is really a **detective task**, not a training task.

---

## 2. The key ideas, in plain words

### DPO (Direct Preference Optimization)
You give the model pairs of answers to the same question:
- **chosen**: the better answer
- **rejected**: the worse answer

DPO nudges the model to make the chosen answer **more likely** and the rejected answer **less likely**. That's all.

### The reference model
DPO measures progress against a **frozen copy of the original model**, called the *reference*. The question it keeps asking is: "compared to where we started, do we now prefer chosen over rejected more?"

**Important detail:** Soup doesn't keep a second copy. It uses the *same* model with the trainable part switched **off**. If that switch ever fails, the "trained" model and the "reference" are identical, and **nothing can be learned**, yet the run still completes.

### LoRA (the adapter)
Retraining a whole model is expensive. LoRA freezes the big model and trains a **small add-on** (about 2 million numbers instead of 1.5 billion). It has two parts, **A** and **B**:
- **B starts at exactly zero.**
- So an untrained adapter changes nothing.
- **If B is still zero (or tiny) after training, the adapter didn't learn.** That's our Check A.

### Layer streaming
Normally the whole model sits in GPU memory. Streaming keeps the model in normal computer RAM and copies it to the GPU **one layer at a time**, like reading a book one page at a time instead of spreading every page on the desk.
- Good: uses very little GPU memory for the model itself.
- Catch: it's new ("BETA") and complicated, so it's where bugs hide.

### Loss
One number that says "how wrong the model is". For DPO it starts at **0.693** (that's ln 2, a coin flip). Going down *usually* means learning, **but not always**:
- the model can make *both* answers less likely, with rejected dropping slightly more
- the model can learn "longer answer = better" instead of "better answer = better"
- the loss can come from something unrelated to what we care about

That's why "loss went down" isn't proof.

---

## 3. What we chose, and why

| Decision | Choice | Why, in one line |
|---|---|---|
| Model | Qwen2.5-1.5B-Instruct | Fits in Colab's 13 GB RAM (streaming needs that) and understands Russian. A 7B model doesn't fit. |
| Data | 600 Russian preference pairs from `saiga_preferences` | Real Russian data. 500 to train, 100 kept aside ("held-out") to test on. |
| Bad data | 63 rows with planted faults, made on purpose | To test whether Soup's data checker notices problems. |
| Precision | bf16 as Soup ships it | The verdict is about Soup *as a user gets it*. |
| Controls | Two deliberately broken runs | A test is only trustworthy if it goes red on something you *know* is broken. |
| Ship rules | Written down **before** any training | So we can't move the goalposts after seeing results. |

### Why the "control" runs matter (the most important idea)
Imagine a smoke detector that never beeps. Is your house safe, or is the detector broken? You only know by **making some smoke on purpose**.

- **lr≈0 run**: training with learning rate basically zero, so the model *should not* change. Our checks must say "NOT trained".
- **Swapped run**: we swap chosen and rejected, so the model learns the *opposite*. Our checks must see it go the wrong way.

If the checks can't tell those apart from the real run, the checks are useless.

---

## 4. The four checks (`scripts/verify_training.py`)

Written and tested *on your laptop first*, with a tiny fake model, before any real training.

| Check | Plain question | What it can't tell you |
|---|---|---|
| **A: Did the adapter change?** | Is part B of LoRA clearly away from zero? | Whether the change is *useful* |
| **B: Does the saved file work?** | Load the saved adapter into a normal model. Did every piece load? Does it change the model's output? | Whether the output is *better* |
| **C: Did it learn on unseen data?** | On the 100 held-out pairs, does the model now prefer chosen over rejected, by more than chance? Did it avoid pushing down *both* answers? Did answers get suspiciously longer? | Whether it's good at things outside this data |
| **D: Do the broken runs look broken?** | Does lr≈0 fail? Does swapped go the opposite way? | Failure types we didn't simulate |

**Why check B exists:** we proved on your laptop that if the saved file's names are slightly wrong, the loading library **loads 0 of 8 pieces and only prints a warning**. You'd be testing the *original* model and think it's the trained one. Soup had exactly this bug in an earlier version.

---

## 5. What we've discovered so far (the good stuff for the report)

### Soup's data checker (`soup data lint`) missed most planted faults
Of 7 kinds of bad data, it fully caught **1** (exact duplicates) and partly caught 1. It missed empty answers, near-duplicates, hidden differences cut off by length limits, and fake look-alike letters. Worse: one check was **skipped** (a missing package) but still displayed **"OK"**.

### Soup's memory predictions disagreed with each other
For the same setup, Soup's tools said **5.5 GB**, **8.0 GB** and **9.95 GB**. My hand calculation said **9.9 GB**.
- The 5.5 GB tool forgot the biggest cost entirely and **recommended a batch size that would crash**.
- The biggest cost is **logits**: the model's score for every word in its 152,000-word vocabulary, at every position. That's **88% of memory**, not the model weights.

### The GPU is running in slow "pretend" mode
The T4 doesn't natively support the number format Soup forces (bf16), so it **emulates** it. Soup's own panel measured **2.4 TFLOPS** where the card can do about **65** in the right format. No warning is shown.

### Soup's streaming + DPO crashed at the very first step
- **What happened:** the "one layer at a time" trick and a second memory-saving trick (from the TRL library, switched on by default) fight each other. During the backward pass (the learning step), the second trick re-runs a layer *outside* the streaming system, finds empty placeholder weights, and crashes.
- **Why:** Soup has a function to switch the second trick off during streaming, but only wired it into regular training, **not into DPO**. We found the exact lines.
- **What we did:** turned the second trick off ourselves (a documented workaround, the same rule Soup already uses elsewhere), and logged it as the #1 defect.
- **The scary part:** just before crashing it printed *"Gradients will be None"*. In a slightly different setup it would **not crash, just silently never learn**. That's exactly the kind of failure the brief cares about most.

### Small things that add up
- Soup silently holds back 10% of the training data (trains on 450, not 500).
- Soup forbids a learning rate of exactly 0, so the "zero" control uses 0.0000000001. That showed our Check A needed a real threshold, not just "is it non-zero".
- The data favours longer answers (72% of the time chosen is longer), so the model may just learn to talk more.

---

## 6. Where AI tools helped, and where they were wrong

Tracked in `AI_USAGE.md`. Two honest examples:
1. The AI **guessed** that Soup's memory check forgot DPO's double batch. Reading the code proved the guess **wrong**.
2. The AI wrote file-reading code that passed its tests but **crashed on real Russian data**: a special invisible "line separator" character inside the text. The tests used simple English-only data.

Lesson for the interview: *tests only prove what they test; real data finds real bugs.*

---

## 7. What happens next

1. **lr≈0 control run** (running now): loss should stay flat at 0.693.
2. **Main run**: 2 passes over the data, about 110 steps.
3. **Swapped control run.**
4. Optional: fp16 run (the "right" number format) and a no-streaming run, to compare speed and memory.
5. Run the 4 checks → verdict.
6. Run `soup ship` too, and compare its opinion with ours.
7. Write the 2-page report.
