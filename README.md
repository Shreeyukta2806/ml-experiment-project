# **ML Experiment Project — Desk Object Classifier**









### **What I Built:**



**A 4-class image classifier trained on a custom dataset I photographed myself — 60 images across four objects: conch shell, diary, matchbox, and a guitar (15 images per class). The goal was to understand the working of the model and observe when does it break and how does the results change when we vary the angle/position/background of the object.**



**I ran 10 controlled experiments, changing one variable at a time, and documented what changed, what surprised me, and what I still don't fully understand.**



**The classes were chosen because they're visually non-trivial — a conch shell and a diary share similar neutral tones, so the model can't just "learn color." It has to learn shape and texture. That made overfitting and generalization visible in a very inevitable way.**



**Architecture: Small custom CNN (3 conv layers + 2 FC layers)**



**Framework: PyTorch | Device: CUDA (RTX 5050)**









### **Experiments Performed:**



**All experiments ran for 15 epochs. One variable changed at a time. The baseline used: Adam optimizer, lr=0.001, batch\_size=16, dropout=0.5 and with no augmentation.**





#### **(Experiment-1):**



##### **Baseline -->**



**Epoch   TrainLoss   TrainAcc   ValLoss  ValAcc**

&#x20;

&#x20;**1       2.1486      25.0%     1.4050    25.0%**



&#x20;**7       0.2463      91.7%     0.8601    58.3%**



&#x20;**15      0.0551      100%      0.4464    75.0%**







**The gap between train accuracy (100%) and val accuracy (75%) by epoch 15 is the prime example of overfitting — the model memorized training images but didn't generalize perfectly to unseen ones. This gap became the reference point for all other experiments.**







#### **(Experiment-2)**

##### **Learning Rate Too Low -->**



**lr=0.00005 (20× slower than baseline)**



**Final val acc: 83.3% — actually better than baseline, but the process was slower. Train loss at epoch 7 (0.61) was roughly where the baseline was at epoch 3. The model was taking tiny steps and needed the full 15 epochs to reach where the baseline was by epoch 8.**



**What I observed: A lower learning rate doesn't mean the model learns less — it means that the model takes much longer, but you're less likely to accidentally jump past the lowest point as in experiment-1, i.e., it prevents overshooting. So, if you train for long enough, the model will still reach a good answer but the rate would be slow. Also, it performed better on images it had never seen before (the validation set), compared to the baseline.**







#### **(Experiment-3)**

##### **Learning Rate Too High (Deliberate Failure) -->**



**lr=1.0 (1000× higher than baseline)**



**Epoch   TrainLoss   Notes**



**1       116,626     Complete divergence from the start**



**6       1,738       Spiked again — loss is erratic**



**9       273         Another spike**



**15      1.3929      Val acc stuck at 25% — random guessing**







**This is the experiment where the loss didn't just stay high — it spiked, partially recovered and spiked again. Val accuracy never got above 41.7% and ended at 25% (pure random guessing for 4 classes).**



**Why this happened:**

**With lr=1.0, each gradient step is so large that the weights overshoot the loss minimum completely. Imagine trying to find the lowest point in a valley by jumping in 10-meter hops — you'll keep flying over it and landing on the other side. The loss oscillates wildly because the optimizer never settles. Adam's adaptive learning rates partially dampens this, which is why the loss eventually started to come down after epoch 7 and due to the damage from early chaotic spikes, the model never learned meaningful features.**









#### **(Experiment-4)**

##### **Batch Size: Small vs Large -->**



**Small batch (4): Final val\_loss=1.3105, val\_acc=67%**



**Large batch (32): Final val\_loss=0.6920, val\_acc=58%**



**Small batch training was noisier — val\_loss spiked late (1.31 at epoch 15 vs 0.38 at epoch 8, suggesting the model started overfitting). Large batch training was smoother but converged slower and ended with lower val\_acc.**



**What I observed: With batch\_size=4 on a 48-image training set, each update sees only 4 samples — the gradient results are noisy and imperfect which can actually help to avoid local minima early on, but causes instability later. Large batches give smoother, more accurate gradient estimates but consider less of the loss landscape per epoch, which is why convergence was slower here.**









#### **(Experiment-5)**

##### **Dropout: Off vs Too High -->**



**No dropout (0.0): train\_acc=100% from epoch 11, train\_loss dropped to 0.0127**



**High dropout (0.7): train\_acc=91.7% at epoch 15, much slower convergence**



**The no-dropout model memorized training data aggressively (train\_loss 0.013 is near-zero), but val accuracy peaked at 83.3% and was unstable. The high-dropout model learned slower but was more stable.**



**What I observed: Without dropout, the models learn to rely on each other's outputs rather than developing independent features. That's why train accuracy hits 100% but generalization suffers. At 0.7 dropout, too many neurons are randomly zeroed each step, making learning slow and noisy. The baseline's 0.5 was quite ideal.**









#### **(Experiment-6)**

##### **SGD Without Momentum -->**



**SGD, no momentum and lr=0.001**



**Final val\_acc=83.3% but train\_loss was still 0.71 at epoch 15 — the model barely converged. Val\_loss was still at 0.88, which is roughly where the baseline was at epoch 5.**



**What I observed: SGD without momentum at the same learning rate as Adam is way more slower. Adam maintains stable learning rates and it effectively has built-in momentum. Vanilla SGD has none of this: every step is purely the current gradient, with no memory of past steps. The model needed far more than 15 epochs to converge.**







#### **(Experiment-7)** 

##### **Weight Decay (L2 Regularization) -->**



**weight\_decay=1e-3**



**Final val\_acc=75.0%, train\_loss=0.1065. Pretty much the same as baseline in terms of accuracy, but the training felt more controlled. No random jumps or dips in the loss.**

**What I observed: Weight decay works by adding a small extra penalty to the loss every time the weights get too large. Basically it tells the model "don't put too much importance on any one thing." On a dataset this small the difference wasn't dramatic — with only 48 training images the model doesn't really have the chance to overfit badly in just 15 epochs anyway — but the training curve was noticeably smoother and more stable compared to baseline.**





#### 

#### **(Experiment-8)**

##### **Data Augmentation (Best Result) -->**



**Random flips, rotation ±15°, color jitter**



**EpochTrain LossTrain AccVal LossVal Acc130.422485.4%0.468891.7%140.474379.2%0.443591.7%150.325087.5%0.402991.7%**

**Val accuracy (91.7%) ended up higher than train accuracy (87.5%) — which is pretty unusual and almost never happens.**



**Why this is the most interesting result: Augmentation takes each training image and randomly flips it, rotates it a bit, and changes the colours slightly every epoch. So the model never sees the exact same photo twice. Because of that it can't just remember what each image looks like — it has to actually figure out what makes an object look like a guitar or a matchbox. Then when it sees the clean, unmodified validation images it actually finds them easier than what it was trained on, which is why val accuracy went higher than train accuracy.**

**With only 60 photos this was by far the biggest improvement I got. If I could only pick one thing to use on a small dataset like this, it would be augmentation.**









#### **What Surprised Me:**



**The learning rate = 1.0 experiment surprised me the most. I expected the loss to just stay high and not really go anywhere. What I didn't expect was how chaotic it was — the loss went from 116K down to 43 by epoch 5, then suddenly jumped back up to 1738 at epoch 6. I thought when training goes unstable it just stays bad. What was actually happening is that Adam was kind of trying to correct itself mid-training, but the weights had already been pushed so far in the wrong direction early on that 15 epochs wasn't enough to fix it.**

**Data augmentation giving higher val acc than train acc genuinely confused me at first. It felt wrong — how is the model doing better on images it's never seen than on the ones it trained on? Then it clicked: the training images were being randomly flipped and rotated and colour shifted every single epoch, so training was actually harder than validation. The validation images were clean and untouched. The model learned to handle messy, varied inputs, so when it saw normal images it did well.**

**SGD without momentum was worse than I expected too. I thought it would just be slower but get to roughly the same place. But after 15 epochs the train loss was still 0.71 — it had barely moved. Adam isn't just a faster version of SGD, it actually works in a different way — it adjusts the learning rate as it goes and remembers which direction it was moving. Plain SGD does none of that.**







#### **Things That Failed or Confused Me:**



**The conch shell folder had a space in the name (conch\_ shell) because of a typo when I made it. PyTorch didn't crash — it just treated it as the class name and kept going — but that made me realise that small mistakes in how you name or organise your data can cause weird problems later, especially if you're feeding class names into something else downstream.**

**I expected weight decay to clearly reduce overfitting compared to baseline but the difference was pretty small. I think it's because with only 48 training images the model doesn't have enough data to go badly overfit in 15 epochs in the first place, so the weight decay didn't have much to fix.**

**I still don't fully understand why the low LR run did slightly better on val accuracy than baseline (83.3% vs 75.0%). My best guess is that taking smaller steps means the model lands in a more stable spot in the loss surface, one that holds up better on new images. But I'm genuinely not sure if that's the real reason or just something that happened because the dataset is so small.**









#### **Insights and Questions I'm Sitting With:** 



**\*If 60 images with augmentation got me to 91.7% accuracy on 4 classes, I'm wondering — how many images would you actually need for something like obstacle detection in a real autonomous system? And does it get harder in a straight line as you add more classes, or does it kind of blow up at some point?**



**\*The lr=1.0 experiment had loss jumping all over the place. I'm curious if there are optimizers that can handle that kind of chaos better — like is there a way to automatically detect when the loss is spiking and dial things back instead of just letting it spiral?**



**\*I didn't test batch normalisation in these experiments, but based on what I saw with large batch size (bs=32 was slow to generalise and took a long time to get anywhere), I think batchnorm might have helped a lot there. That's something I want to try next.**



**\*For real use cases like obstacle detection or anything running in a ROS pipeline — is there a proper way to figure out a starting learning rate, or is it always just trial and error? Doing a full grid search every time seems way too slow for anything practical.**





