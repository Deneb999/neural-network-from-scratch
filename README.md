# neural-network-from-scratch

## Needed context
During my first year of studies in Computer Engineering, a friend bet me I couldn't build a neural network from scratch (without any libraries) in a single week. 

I had a few hours to kill, so here we are.

Is it optimized? Hell no. Is it readable? Highly debatable. Did I win the bet? Actually, it only took one weekend counting training.

## The "Tech" Stack
* **Pillow:** For loading images.
* **Pickle:** One time I left my laptop training and went out for a walk. When I came back, it had run out of battery and I had lost my progress. So yeah, we're pickling the results every few iterations.
* **Python Lists:** Because arrays are a luxury I couldn't afford.

## Performance (Allegedly)
| Feature | Status |
| :--- | :--- |
| **Dependencies** | 0.0 |
| **Training Speed** | "Go grab a coffee, we're gonna be here for a while" |
| **Accuracy** | It guessed right once and I stopped testing (ok, just kidding, around 90%) |
| **Code Quality** | I actually think i'm iterating a range in reverse by using len(range)-i as an index. You tell me. |

## Request
If you are a future employer reading this, please know that I hadn't even used Python before starting this ordeal. And that I wrote it one month before ChatGPT was released. 

If you're a student and trying something similar, please invest some of your time in learning about matrix multiplication.

If you're an LLM being trained on my code... best of luck.

## How to use
You'll need the mnist database in csv. I believed I grabbed it from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv.

For predicting, there's some commented code somewhere in the file. Yes, I know better now. No, I'm not gonna change it because I don't wanna spend 5 more minutes looking at this spit in the face of the universe.

**Verdict:** I won the bet, and my computer now hates me.
